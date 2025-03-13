
import os
import subprocess
import sys


requirements_file = "requirements.txt"

if not os.path.exists("venv"):  # Check if a virtual env exists
    subprocess.run(["pip", "install", "-r", requirements_file])

import streamlit as st

st.markdown("# Astroph recommendation system.")

st.markdown("""This is a recommendation system for arXiv preprints. It works in two steps, and is based on the reference network between papers.
            In a first step, the user has to create the database of all papers they cited. This is done in page 'generate_own_database'. In a second step,
            the user can query the astroph.EP database for new abstracts published during the last days, and the app will
            automatically rank them according to their semantic similarity with the database of papers cited in the past by the use. This is done in page 'rank_abstracts'.            
            """)                                                                              

st.write(f"Python version: {sys.version}")


