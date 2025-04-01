"""
Home page for the Astroph recommendation system.
This module handles the main landing page and initial setup of the application.
"""

import os
import subprocess
import sys
import streamlit as st

# Constants
REQUIREMENTS_FILE = "requirements.txt"
VENV_DIR = "venv"

def setup_virtual_environment():
    """Set up the virtual environment and install requirements if needed."""
    if not os.path.exists(VENV_DIR):
        try:
            subprocess.run(["pip", "install", "-r", REQUIREMENTS_FILE], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install requirements: {e}")
            return False
    return True

def main():
    """Main function to render the home page."""
    # Set page config
    st.set_page_config(
        page_title="Astroph Recommendation System",
        page_icon="📚",
        layout="wide"
    )
    
    # Setup virtual environment
    if not setup_virtual_environment():
        return

    # Main content
    st.title("Astroph Recommendation System")
    
    st.markdown("""
    This is a recommendation system for arXiv preprints. It works in two steps, and is based on the reference network between papers.

    ### How it works:
    1. **Create Your Database**: 
       - Go to the 'Generate Own Database' page
       - Create a database of papers you've cited

    2. **Get Recommendations**:
       - Visit the 'Rank Abstracts' page
       - Query the astroph.EP database for new abstracts
       - Get automatically ranked results based on semantic similarity with your citation database
    """)

    # System information
    with st.expander("System Information"):
        st.write(f"Python version: {sys.version}")

if __name__ == "__main__":
    main()


