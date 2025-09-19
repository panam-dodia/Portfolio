import sys
from pathlib import Path
import streamlit as st

# Get the base directory
BASE_DIR = Path(__file__).parent

# Add all project directories to Python path
sys.path.append(str(BASE_DIR / "views" / "projects" / "blog"))
sys.path.append(str(BASE_DIR / "views" / "projects" / "pdf_QA"))
sys.path.append(str(BASE_DIR / "views" / "projects" / "document_structured"))
sys.path.append(str(BASE_DIR / "views" / "projects" / "english_german"))

# --- PAGE SETUP ---
about_page = st.Page(
    str(BASE_DIR / "views" / "about_me.py"),
    title="About Me",
    icon=":material/account_circle:",
    default=True,
    url_path="about"
)

project_1_page = st.Page(
    str(BASE_DIR / "views" / "projects" / "blog" / "blog_app.py"),
    title="Generate a Blog",
    icon=":material/bar_chart:",
    url_path="blog"
)

project_2_page = st.Page(
    str(BASE_DIR / "views" / "projects" / "pdf_QA" / "document_QA.py"),
    title="Chat with the Document",
    icon=":material/smart_toy:",
    url_path="pdf-qa"
)

project_3_page = st.Page(
    str(BASE_DIR / "views" / "projects" / "document_structured" / "doc_app.py"),
    title="Clean and Structure your Document",
    icon=":material/sentiment_very_satisfied:",
    url_path="document-structured"
)

project_4_page = st.Page(
    str(BASE_DIR / "views" / "projects" / "english_german" / "translation_app.py"),
    title="Translate from English to German",
    icon=":material/image_search:",
    url_path="english-german"
)

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "Live Projects": [project_3_page, project_2_page, project_4_page, project_1_page],
    }
)

# --- RUN NAVIGATION ---
pg.run()