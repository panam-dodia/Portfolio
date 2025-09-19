import os
import streamlit as st
from PIL import Image
from pathlib import Path

# --- SETTINGS ---
PAGE_TITLE = "Panam Dodia | AI Engineer"
PAGE_ICON = "🤖"
NAME = "Panam Dodia"
DESCRIPTION = "Aspiring AI/ML Engineer | Skilled in LLMs, Big Data, and Cloud"
EMAIL = "panamdodia945@gmail.com"
SOCIALS = {
    "LinkedIn": "https://www.linkedin.com/in/panamdodia/",
    "GitHub": "https://github.com/panam-dodia"
}

resume_file = "assets/Panam_Dodia_AI_Engineer.pdf"
profile_pic = "assets/Profile_Pic.png"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- HERO ---
col1, col2 = st.columns([1, 3], gap="medium")
with col1:
    st.image(profile_pic, width=200)
with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.download_button("📄 Download Resume", open(resume_file, "rb").read(),
                       file_name="Panam_Dodia_Resume.pdf")
    st.write(f"📫 {EMAIL}")
    cols = st.columns(len(SOCIALS))
    for (platform, link), col in zip(SOCIALS.items(), cols):
        col.markdown(f"[{platform}]({link})")

st.markdown("---")

# --- SKILLS ---
st.subheader("🛠️ Skills")
skills = {
    "Programming": ["Python", "SQL", "Matlab"],
    "ML Libraries": ["Scikit-learn", "TensorFlow", "PyTorch", "LangChain"],
    "Big Data": ["Hadoop", "Spark", "Cassandra", "Hive", "HDFS"],
    "Cloud": ["AWS", "Azure", "GCP", "SageMaker"],
    "BI Tools": ["Excel", "Power BI", "Tableau"],
    "Databases": ["MySQL", "SQLite"]
}
cols = st.columns(3)
for i, (area, items) in enumerate(skills.items()):
    with cols[i % 3]:
        st.markdown(f"**{area}**")
        st.write(", ".join(items))

st.markdown("---")

# --- EXPERIENCE ---
st.subheader("💼 Professional Experience")
experience = [
    {
        "role": "AI Engineer",
        "company": "TnS10X.ai, Denton, TX",
        "date": "Aug 2025 – Sep 2025",
        "details": [
            "Developed Gmail add-on with GPT-3.5 achieving 95% accuracy.",
            "Reduced unwanted emails by 90% using pattern recognition.",
            "Deployed full-stack solution processing 30K+ emails monthly."
        ]
    },
    {
        "role": "Research Assistant",
        "company": "University of North Texas",
        "date": "May 2024 – Present",
        "details": [
            "Integrated DaCLIP with Sparse Autoencoder for image restoration (+2.89 PSNR).",
            "Developed novel loss functions improving robustness to degradations."
        ]
    },
    {
        "role": "Machine Learning Engineer",
        "company": "AdTech, India",
        "date": "Jan 2022 – Jun 2023",
        "details": [
            "Deployed ML models improving accuracy by 18%.",
            "CI/CD pipelines on AWS reduced deployment time by 60%.",
            "Managed EC2, S3, SageMaker for scalable ML workflows."
        ]
    }
]
for job in experience:
    st.markdown(f"**{job['role']} | {job['company']}**")
    st.caption(job["date"])
    for d in job["details"]:
        st.write(f"- {d}")
    st.write("")

st.markdown("---")

# --- PROJECTS ---
st.subheader("📂 Projects")
projects = {
    "Intelligent Document RAG System": "Built with LlamaIndex & OpenAI; 42% faster query response, 93% accuracy.",
    "Spiritual AI Chatbot": "Full-stack chatbot (FastAPI + GPT-4 + ChromaDB) with self-harm prevention.",
    "Predictive Analytics for Sales": "Spark + Hive pipeline with 91% accurate forecasting.",
    "English to German Translator": "MarianMT + Streamlit app, 99.9% BLEU score (from ML resume)."
}
for name, desc in projects.items():
    with st.expander(name):
        st.write(desc)

st.markdown("---")

# --- EDUCATION ---
st.subheader("🎓 Education")
st.write("**M.S. in Artificial Intelligence** — University of North Texas (2024–2025, GPA 4.0)")
st.write("**B.C.A.** — Veer Narmad South Gujarat University (2018–2021, GPA 3.2)")