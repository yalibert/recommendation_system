"""
Generate Own Database Page
This module handles the creation of a personalized paper database based on author citations.
It uses the ADS API to fetch papers and their references, computes embeddings using SciBERT,
and generates a weighted database of cited papers.
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
import ads
import tqdm
import time
from typing import List, Dict, Tuple, Optional

# Constants
MODEL_PATH = "model_checkpoint.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/i03kzt85quppl8ka64c4i/model_checkpoint.pth?rlkey=u53srsucm0jwmtd2xfprpzgv3&dl=1"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 50
MAX_RETRIES = 3

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

    # Tokenize texts
    tokenized_texts = [
        tokenizer.encode(
            text,
            padding='max_length',
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).squeeze() for text in list_abstracts
    ]

    # Create attention masks
    att_masks = torch.tensor([
        [int(element > 0) for element in ids] + [0] * (MAX_SEQUENCE_LENGTH - len(ids))
        for ids in tokenized_texts
    ])

    # Stack and move to device
    tokenized_texts = torch.stack(tokenized_texts).to(device)
    att_masks = att_masks.to(device)

    # Compute embeddings
    with torch.no_grad():
        embeddings = model(tokenized_texts, attention_mask=att_masks).last_hidden_state[:, 0]

    return embeddings.cpu().numpy()

def compute_weights(papers_by_author: List[ads.Article], author_list: List[str]) -> List[float]:
    """
    Compute weights for papers based on author position in the author list.
    
    Args:
        papers_by_author: List of papers from ADS
        author_list: List of author names to consider
        
    Returns:
        List of weights for each paper
    """
    weight_paper = []
    
    for paper in papers_by_author:
        authors = paper.author
        num_authors = len(authors)
        weight = 0
        
        author_names = list(set(author_list) & set(authors))
        
        if author_names:
            author_name = author_names[0]
            position = authors.index(author_name) + 1
            weight = 1. - (position-1)/num_authors
        else:
            st.warning(f"Author not found in paper: {paper.title}")
        
        weight_paper.append(weight)
    
    return weight_paper

def make_all_papers_dict(papers_by_author: List[ads.Article], weight_paper: List[float]) -> Dict[str, float]:
    """
    Create a dictionary of all cited papers with their weights.
    
    Args:
        papers_by_author: List of papers from ADS
        weight_paper: List of weights for each paper
        
    Returns:
        Dictionary mapping paper bibcodes to their weights
    """
    all_papers_dict = {}
    
    for paper, weight in zip(papers_by_author[1:], weight_paper[1:]):
        if paper.reference is not None:
            for ref in paper.reference:
                all_papers_dict[ref] = all_papers_dict.get(ref, 0) + weight
    
    st.info(f"Number of cited papers: {len(all_papers_dict)}")
    return all_papers_dict

def download_and_embed_papers(all_papers_dict: Dict[str, float]) -> Dict[str, dict]:
    """
    Download and compute embeddings for papers in the dictionary.
    
    Args:
        all_papers_dict: Dictionary mapping bibcodes to weights
        
    Returns:
        Dictionary containing paper information and embeddings
    """
    bibcodes = list(all_papers_dict.keys())
    weights = list(all_papers_dict.values())
    
    # Create batches
    chunk_batch_list = [bibcodes[i:i+BATCH_SIZE] for i in range(0, len(bibcodes), BATCH_SIZE)]
    weight_batch_list = [weights[i:i+BATCH_SIZE] for i in range(0, len(weights), BATCH_SIZE)]
    
    dict_papers = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for counter, (bibcode_chunk, weight_chunk) in enumerate(zip(chunk_batch_list, weight_batch_list)):
        status_text.text(f"Processing batch {counter + 1}/{len(chunk_batch_list)}")
        progress_bar.progress((counter + 1) / len(chunk_batch_list))
        
        query_string = " OR ".join(f"bibcode:{bc}" for bc in bibcode_chunk)
        
        for attempt in range(MAX_RETRIES):
            try:
                reference_papers = list(ads.SearchQuery(
                    q=query_string,
                    fl=['title', 'abstract', 'pubdate', 'bibcode', 'author', 'reference']
                ))
                break
            except Exception as e:
                st.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(120)
                if attempt == MAX_RETRIES - 1:
                    st.error(f"Failed to retrieve data after {MAX_RETRIES} attempts")
                    continue
        
        list_abstracts = []
        for paper in reference_papers:
            if paper.abstract is not None:
                list_abstracts.append(paper.abstract)
            else:
                st.warning(f"No abstract found for paper: {paper.title}")
                continue
        
        if list_abstracts:
            embeddings = compute_embeddings(list_abstracts)
            
            for paper, embedding, weight in zip(reference_papers, embeddings, weight_chunk):
                dict_papers[paper.bibcode] = {
                    'title': paper.title,
                    'abstract': paper.abstract,
                    'pubdate': paper.pubdate,
                    'author': paper.author,
                    'reference': paper.reference,
                    'weight': weight,
                    'embedding': embedding.tolist()
                }
    
    return dict_papers

def get_author_papers(author_list: List[str], ads_api_key: str) -> List[ads.Article]:
    """
    Fetch papers for a list of authors using the ADS API.
    
    Args:
        author_list: List of author names
        ads_api_key: ADS API key
        
    Returns:
        List of papers from ADS
    """
    ads.config.token = ads_api_key
    
    papers_by_author = []
    for author in author_list:
        papers = list(ads.SearchQuery(
            author=author,
            fl=['title', 'abstract', 'pubdate', 'bibcode', 'author', 'reference']
        ))
        papers_by_author.extend(papers)
    
    return papers_by_author

def main():
    """Main function to render the page and handle user interactions."""
    st.set_page_config(
        page_title="Generate Own Database",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Generate Own Database")
    
    st.markdown("""
    ### How to Use This App
    1. Enter author names **separated by `|`. Be careful to enter all possible variations of your name
    2. Input your **ADS API key** (kept private).
    3. Click **"Search Papers"** to fetch results.
    4. Download the results as a JSON file.
    5. Monitor progress with the progress bar.

    This app queries the **ADS API** and retrieves all papers from the author, and all paper cited by these papers.
    It then assign a weight to each of these cited papers (based on the position of the author in the author list), and downloads
    the abstracts of these cited papers. Finally, the sciBERT model, fine-tuned to optimze citation prediction, is used to
    compute the embeddings of all these cited papers.

    The result is a json file that contains, for each cited paper, the abstract, the publication date,
    the title, the authors, the references, the weight and the embedding. 
                
    This file can be downloaded and stored. It can be used later on to evaluate the similarity of
    new papers with this database, assuming that if a new paper is similar, in term of its abstract,
    to many papers cited by the author, it is likely to be interesting for the author.
                
    This ranking of new papers is done in page 'rank new papers'.
    """)
    
    # Load model
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer, st.session_state.model, st.session_state.device = load_model()
    
    # User input
    author_input = st.text_input("Enter author names (separated by |)")
    ads_api_key = st.text_input("Enter your ADS API key", type="password")
    
    if st.button("Search Papers") and author_input and ads_api_key:
        author_list = [name.strip() for name in author_input.split("|")]
        
        with st.spinner("Fetching papers..."):
            papers_by_author = get_author_papers(author_list, ads_api_key)
            
            if not papers_by_author:
                st.error("No papers found for the given authors.")
                return
            
            st.success(f"Found {len(papers_by_author)} papers.")
            
            # Compute weights and create paper dictionary
            weight_paper = compute_weights(papers_by_author, author_list)
            all_papers_dict = make_all_papers_dict(papers_by_author, weight_paper)
            
            # Download and embed papers
            dict_papers = download_and_embed_papers(all_papers_dict)
            
            # Save results
            json_str = json.dumps(dict_papers)
            st.download_button(
                label="Download Database",
                data=json_str,
                file_name="paper_database.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()

