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

st.title("Generate own database")


# Explanation text
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
# --------------------------------------------
# 0. Settings and Suppress Transformers Warnings
# --------------------------------------------

logging.set_verbosity_error()  # Suppress Hugging Face warnings

# --------------------------------------------
# 1. Download Model Weights from Dropbox if Needed
# --------------------------------------------
MODEL_PATH = "model_checkpoint.pth"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/i03kzt85quppl8ka64c4i/model_checkpoint.pth?rlkey=u53srsucm0jwmtd2xfprpzgv3&dl=1"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights from Dropbox...")
        response = requests.get(DROPBOX_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
        else:
            st.error("Failed to download model weights.")
            return False
    return True

# --------------------------------------------
# 2. Load Model (only once) using session_state caching
# --------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    # Download model weights if needed
    if not download_model():
        st.stop()  # Stop execution if model download fails

    # Initialize tokenizer and model.
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_cased', do_lower_case=True)
    model_full = BertForSequenceClassification.from_pretrained(
        'allenai/scibert_scivocab_cased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    # We only need the underlying BERT encoder for embeddings.
    model = model_full.bert
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the checkpoint weights (using strict=False to bypass missing classifier weights)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return tokenizer, model, device

# Load and cache the model in session_state
#if "tokenizer" not in st.session_state:
st.session_state.tokenizer, st.session_state.model, st.session_state.device = load_model()
   
# --------------------------------------------
# 📌 3. Function to Compute Embeddings
# --------------------------------------------

def compute_embeddings(list_abstracts):
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    device = st.session_state.device
    max_sequence_length = 512


# Tokenize each inner list of texts
    tokenized_texts = [tokenizer.encode(text, padding='max_length', max_length=max_sequence_length,truncation=True, return_tensors="pt").squeeze() for text in list_abstracts]

# Pad each tensor in the list to ensure they are all the same length
    padded_texts = [
        torch.cat([tensor, torch.zeros(max_sequence_length - tensor.size(0), dtype=torch.long)]) for tensor in tokenized_texts
    ]

    tokenized_texts = torch.stack(tokenized_texts)
    #print(tokenized_texts.shape)  # Should print (batch_size, 3, m
# attention masks    
    att_masks = []
    
    for ids in tokenized_texts:
        masks = [int(element > 0) for element in ids]
        att_masks.append(masks)
        
    padded_lists = [[mask + [0] * (max_sequence_length - len(mask)) for mask in att_masks]]
    att_masks = np.array(padded_lists).squeeze()
    att_masks = att_masks.reshape(1, -1) if att_masks.ndim == 1 else att_masks
    att_masks = torch.Tensor(att_masks)
# computing the embeddings
    text = tokenized_texts
    mask = att_masks
    embedding = model(text, attention_mask=mask).last_hidden_state[:, 0]  

    return embedding.detach().numpy()

# --------------------------------------------
# 📌 . Computing weights of papers and all_papers_dict
# --------------------------------------------

def compute_weights(papers_by_author):
    weight_paper = []

    for paper in papers_by_author:
        authors = paper.author
        num_authors = len(authors)
        weight = 0
    
        author_names = list(set(author_list) & set(authors))

    # Check if Author is in the author list
        if len(author_names) > 0 :
            author_name = author_names[0]
            position = authors.index(author_name) + 1  # Position is 1-based index
            weight = 1. - (position-1)/num_authors
        else:
            st.write(f"Author not found in the author list of paper: {paper.title}")
  
        weight_paper.append(weight)  

    return weight_paper   

def make_all_papers_dict(papers_by_author):   

    all_papers_dict = {}

    for paper,weight in zip(papers_by_author[1:],weight_paper[1:]):
        if paper.reference is not None:
            print(paper.bibcode,paper.title)
            print(paper.reference)
            for ref in paper.reference:
                if ref in all_papers_dict:
                    all_papers_dict[ref] += weight
                else:
                    all_papers_dict[ref] = weight

    st.write("Number of cited papers:",len(all_papers_dict))

    return all_papers_dict

# --------------------------------------------
# 📌 . Compute and embed papers
# --------------------------------------------

def download_and_embed_papers(all_papers_dict):
    abstract_list = []
    title_list = []
    weight_list = []
    embeddings_list = []
    date_list = []
    bibcode_list = []
    author_list = []
    reference_list = []

    bibcodes = list(all_papers_dict.keys())
    weights = list(all_papers_dict.values())

    # Define the batch size and retry limit
    batch_size = 50
    max_retries = 3

    chunk_batch_list = []
    weight_batch_list = []
    for i in range(0,len(bibcodes),batch_size):
        chunk_batch_list.append(bibcodes[i:i+batch_size])
        weight_batch_list.append(weights[i:i+batch_size])

    # downloading the abstracts

    dict_papers = {}

    # Create a progress bar in Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()  # To display current progress

    total_chunks = len(chunk_batch_list)

    # Iterate over chunks with a progress bar
    for counter, (bibcode_chunk, weight_chunk) in enumerate(zip(chunk_batch_list, weight_batch_list)):
        # Simulate processing time (replace with actual processing)
        
        # Join bibcodes with 'OR' to form a single query string
        query_string = " OR ".join(f"bibcode:{bc}" for bc in bibcode_chunk)
        #print(query_string)
        for attempt in range(max_retries):
            try:
                # Query ADS for the current batch
                reference_papers = list(ads.SearchQuery(q=query_string, fl=['title', 'abstract', 'pubdate', 'bibcode','author','reference']))
                #for paper in reference_papers:
                #    print(paper.bibcode)
                # Optional: Print progress or the number of papers added
                #st.write(f"Retrieved {len(reference_papers)} papers in this batch - {len(bibcode_chunk)} expected")
                break  # Break out of retry loop if successful

            except Exception as e:
                st.write(f"Attempt {attempt + 1} failed for current batch. Error: {e}")
                time.sleep(120)  # Delay before retrying
                
                # If max attempts reached, log the failure
                if attempt == max_retries - 1:
                    st.write(f"Failed to retrieve data for this batch after {max_retries} attempts.")
        
        list_abstracts = []
        for i,paper in enumerate(reference_papers):
            if paper.abstract is not None:
                list_abstracts.append(paper.abstract)
            else:
                list_abstracts.append("there is no abstract in this paper")

        embeddings = compute_embeddings(list_abstracts)

        # Update progress bar
        progress = (counter + 1) / total_chunks
        progress_bar.progress(progress)

        # Update status text
        status_text.text(f"Processing {counter+1}/{total_chunks} chunks...")

    # Check if the paper was found and print the abstract
        for i,paper in enumerate(reference_papers):
            if paper.abstract is not None:
                dict_papers[paper.bibcode] = {
#                    'abstract' : paper.abstract,
                    'date' : paper.pubdate,
                    'title' : paper.title,
                    'authors' : paper.author,
                    'references' : paper.reference,
                    'weight' : weight_chunk[i],
                    'embedding' : embeddings[i]
                }
            else:
                print(f"No paper found for Bibcode: {paper.bibcode}")
                print(f"Title: {paper.title}",paper.bibcode)

        # Add delay between batches to avoid overwhelming the API
        time.sleep(5)

    # Optional: Completion message
    status_text.text("Processing complete! ✅")
    st.success("All chunks processed successfully!")

    for bibcode, dict_paper in dict_papers.items():
        dict_paper["embedding"] = dict_paper["embedding"].tolist()

    # Save dict to a JSON file
    json_data = json.dumps(dict_papers, indent=4)

    return dict_papers,json_data

# --------------------------------------------
# 📌 . Getting papers by author
# --------------------------------------------

def get_author_papers(author_list,ads_api_key):
# Initialize the ADS API key
    ads.config.token = ads_api_key  
        # Generate the OR-separated query for multiple authors
    author_query = " OR ".join([f'author:"{name}"' for name in author_list])
    full_query = f"({author_query}) AND property:refereed"

    st.write(author_query)

# Perform the search with the updated query
    papers_by_author = list(ads.SearchQuery(
                q=full_query, 
                fl=['id', 'title', 'author', 'abstract', 'reference', 'bibcode'], 
                rows=2000
    ))          

    filtered_papers = [p for p in papers_by_author if any(name in p.author for name in author_list)]

    st.write(f"Found {len(filtered_papers)} papers.")
    
    return filtered_papers

# --------------------------------------------
# 📌 5. User Input
# --------------------------------------------

# Step 1: Get author names input (split by |)
author_input = st.text_input("Enter author names (separated by '|'):", "Alibert, Y. |  Alibert, Yann")
author_list = [name.strip() for name in author_input.split("|") if name.strip()]

# Step 2: Get ADS API key (hidden)
ads_api_key = st.text_input("Enter your ADS API Key:", type="password")

# Step 3: Search ADS API when user clicks the button

if st.button("Search Papers"):
    if not ads_api_key:
        st.error("Please enter your ADS API key.")
    elif not author_list:
        st.error("Please enter at least one author name.")
    else:
        st.write('in dev')
        papers_by_author = get_author_papers(author_list,ads_api_key)
        weight_paper = compute_weights(papers_by_author)
        all_papers_dict = make_all_papers_dict(papers_by_author)
        st.write('Now downloading all abstract cited. This will take a while...')
        dict_papers,json_data = download_and_embed_papers(all_papers_dict)
        # Provide a download button
        st.download_button(
            label="Download JSON file",
            data=json_data,
            file_name="dict_papers_cited_by_author.json",
            mime="application/json"
        )

