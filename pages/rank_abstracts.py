"""
Rank Abstracts Page
This module handles the ranking of new arXiv abstracts based on their similarity to a user's citation database.
It uses the SciBERT model to compute embeddings and cosine similarity to rank papers.
"""

import streamlit as st
import torch
import requests
import os
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, logging
import io
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional

# Constants
MODEL_PATH = "model_checkpoint.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/i03kzt85quppl8ka64c4i/model_checkpoint.pth?rlkey=u53srsucm0jwmtd2xfprpzgv3&dl=1"
MAX_SEQUENCE_LENGTH = 512
MAX_RESULTS = 200

# --------------------------------------------
# 0. Settings and Suppress Transformers Warnings
# --------------------------------------------
logging.set_verbosity_error()

def download_model() -> bool:
    """
    Download model weights from Dropbox if they don't exist locally.
    
    Returns:
        bool: True if download successful or model exists, False otherwise
    """
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights from Dropbox...")
        try:
            response = requests.get(DROPBOX_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
            return True
        except requests.RequestException as e:
            st.error(f"Failed to download model weights: {e}")
            return False
    return True

@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[BertTokenizer, torch.nn.Module, torch.device]:
    """
    Load and initialize the SciBERT model and tokenizer.
    
    Returns:
        Tuple containing (tokenizer, model, device)
    """
    if not download_model():
        st.stop()

    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_cased', do_lower_case=True)
    model_full = BertForSequenceClassification.from_pretrained(
        'allenai/scibert_scivocab_cased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model = model_full.bert
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

def fetch_recent_arxiv_abstracts(domain: str = "astro-ph.EP", days: int = 1) -> Tuple[List[str], List[str], List[str]]:
    """
    Fetch recent abstracts from arXiv API.
    
    Args:
        domain: arXiv category to search in
        days: Number of days to look back
        
    Returns:
        Tuple of (abstracts, titles, links)
    """
    base_url = "http://export.arxiv.org/api/query"
    
    params = {
        "search_query": domain,
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Failed to fetch data from arXiv: {e}")
        return [], [], []
    
    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    date_limit = datetime.now() - timedelta(days=days)
    
    abstracts, titles, links = [], [], []
    
    for entry in root.findall("atom:entry", ns):
        published_date = datetime.fromisoformat(entry.find("atom:published", ns).text.strip()[:-1])
        
        if published_date >= date_limit:
            title = entry.find("atom:title", ns).text.strip()
            abstract = entry.find("atom:summary", ns).text.strip()
            link = entry.find("atom:id", ns).text.strip()
            
            abstracts.append(abstract)
            titles.append(title)
            links.append(link)
    
    return abstracts, titles, links

def compute_embeddings(list_abstracts: List[str]) -> np.ndarray:
    """
    Compute embeddings for a list of abstracts using the SciBERT model.
    
    Args:
        list_abstracts: List of abstract texts to embed
        
    Returns:
        numpy.ndarray: Array of embeddings
    """
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    device = st.session_state.device

    tokenized_texts = [
        tokenizer.encode(
            text,
            padding='max_length',
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).squeeze() for text in list_abstracts
    ]
    
    tokenized_texts = torch.stack(tokenized_texts)
    att_masks = (tokenized_texts > 0).float()
    tokenized_texts, att_masks = tokenized_texts.to(device), att_masks.to(device)
    
    with torch.no_grad():
        embeddings = model(tokenized_texts, attention_mask=att_masks).last_hidden_state[:, 0]
    
    return embeddings.cpu().numpy()

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an array to range [0, 1].
    
    Args:
        arr: Input array
        
    Returns:
        Normalized array
    """
    return (arr - arr.min(axis=1, keepdims=True)) / (arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))

def format_title(text: str, width: int = 20) -> str:
    """
    Format title text with line breaks.
    
    Args:
        text: Title text
        width: Maximum width per line
        
    Returns:
        Formatted text with line breaks
    """
    return "\n".join(textwrap.wrap(text, width))

def rank_by_weighted_cosine_similarity(
    ref_embeddings: np.ndarray,
    recent_abstracts_embeddings: np.ndarray,
    ref_weights: np.ndarray,
    aggregation: str = "mean"
) -> List[Tuple[int, float]]:
    """
    Rank recent abstracts based on weighted cosine similarity with reference embeddings.
    
    Args:
        ref_embeddings: Reference paper embeddings
        recent_abstracts_embeddings: New paper embeddings
        ref_weights: Weights for reference papers
        aggregation: Method to aggregate similarities ("max", "mean", or "sum")
        
    Returns:
        List of (index, score) tuples sorted by similarity
    """
    similarity_matrix = cosine_similarity(recent_abstracts_embeddings, ref_embeddings)
    weighted_similarity_matrix = similarity_matrix * ref_weights
    
    if aggregation == "max":
        scores = np.max(weighted_similarity_matrix, axis=1)
    elif aggregation == "mean":
        scores = np.mean(weighted_similarity_matrix, axis=1)
    elif aggregation == "sum":
        scores = np.sum(weighted_similarity_matrix, axis=1)
    else:
        raise ValueError("Invalid aggregation method. Choose 'max', 'mean', or 'sum'.")
    
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

def main():
    """Main function to render the page and handle user interactions."""
    st.set_page_config(
        page_title="Rank Abstracts",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Rank arXiv Abstracts")
    
    # Load model
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer, st.session_state.model, st.session_state.device = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload the database JSON file", type="json", key="database_json")
    
    if uploaded_file is not None:
        try:
            data = uploaded_file.getvalue().decode("utf-8")
            dict_papers = json.loads(data)
            st.success("File successfully uploaded and read.")
            
            ref_embeddings = np.array([dict_papers[bibcode]['embedding'] for bibcode in dict_papers.keys()])
            weights = np.array([dict_papers[bibcode]['weight'] for bibcode in dict_papers.keys()])
            
            days = st.number_input(
                "Number of days to fetch abstracts",
                min_value=0,
                max_value=100,
                step=1,
                value=10,
                format="%d"
            )
            
            if days > 0:
                with st.spinner(f"Fetching abstracts from the last {days} days..."):
                    abstracts, titles, links = fetch_recent_arxiv_abstracts(days=days)
                
                if abstracts:
                    st.success(f"Found {len(abstracts)} abstracts.")
                    
                    with st.spinner("Computing embeddings..."):
                        embeddings = compute_embeddings(abstracts)
                    
                    rankings = rank_by_weighted_cosine_similarity(
                        ref_embeddings,
                        embeddings,
                        weights,
                        aggregation="max"
                    )
                    
                    # Display rankings
                    for index, score in rankings:
                        with st.expander(f"📄 {titles[index]} (Score: {score:.3f})"):
                            st.markdown(f"**Abstract:**\n{abstracts[index]}")
                            st.markdown(f"[Open PDF]({links[index].replace('/abs/', '/pdf/')})")
                else:
                    st.warning("No abstracts found for the specified time period.")
            
            elif days == 0:
                st.info("Exited. Refresh the page to start again.")
                
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid database file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

