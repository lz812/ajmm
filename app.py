import os


# ========== INSTALL ==========
# Run this in your terminal first:
# pip install streamlit sentence-transformers requests

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import requests
import csv
import pandas as pd

JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")

if not JSEARCH_API_KEY:
    st.error("❌ Missing API key. Please set JSEARCH_API_KEY in Streamlit Secrets.")
    st.stop()

# ========== Streamlit UI ==========
st.set_page_config(page_title="AI Job Matcher", layout="centered")

st.markdown("""
    <style>
    body {
        font-family: 'Helvetica Neue', sans-serif;
        background-color: #ffffff;
        color: #111;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #111;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🔍 AI-Powered Job Matcher</h1>', unsafe_allow_html=True)

resume_input = st.text_area("📄 Paste your Resume Text here", height=300)
job_keywords_input = st.text_area("🎯 Desired Job Titles (comma-separated)", height=100)
generate = st.button("🚀 Generate Matches")

# ========== Constants ==========
LOCATIONS = ["United States", "Remote United States"]
SIMILARITY_THRESHOLD = 60.0
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# ========== Helpers ==========
def clean_job_text(text):
    lines = text.split("\n")
    return " ".join([
        line for line in lines if any(kw in line.lower() for kw in ["responsib", "require", "skill", "qualif"]) or len(line) > 60
    ])

def best_chunk_score(chunks, job_text, model):
    try:
        job_emb = model.encode(job_text, convert_to_tensor=True)
        scores = [
            util.pytorch_cos_sim(model.encode(chunk, convert_to_tensor=True), job_emb)[0][0]
            for chunk in chunks
        ]
        return round(float(max(scores)) * 100, 2)
    except:
        return 0.0

def fetch_jobs(keywords, locations):
    headers = {
        "X-RapidAPI-Key": JSEARCH_API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    jobs = []
    for keyword in keywords:
        for location in locations:
            params = {
                "query": f"{keyword} in {location}",
                "page": "1",
                "num_pages": "1",
                "date_posted": "week",
                "employment_types": "intern,fulltime",
            }
            try:
                resp = requests.get("https://jsearch.p.rapidapi.com/search", headers=headers, params=params)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    for job in data:
                        jobs.append({
                            "title": job.get("job_title", ""),
                            "company": job.get("employer_name", ""),
                            "description": job.get("job_description", ""),
                            "link": job.get("job_apply_link", ""),
                            "date": job.get("job_posted_at_datetime_utc", ""),
                            "source_keyword": keyword,
                            "location": location
                        })
            except:
                pass
    return jobs

# ========== Main App ==========
if generate:
    if not resume_input.strip() or not job_keywords_input.strip():
        st.warning("Please enter both resume text and job titles.")
    else:
        with st.spinner("🧠 Loading model and fetching jobs..."):
            model = SentenceTransformer(MODEL_NAME)
            resume_chunks = [chunk.strip() for chunk in resume_input.split("\n\n") if chunk.strip()]
            search_keywords = [kw.strip() for kw in job_keywords_input.split(",") if kw.strip()]
            jobs = fetch_jobs(search_keywords, LOCATIONS)

        st.info(f"🔍 {len(jobs)} jobs fetched. Scoring now...")

        matched_jobs = []
        for job in jobs:
            job_text = clean_job_text(job["description"])
            score = best_chunk_score(resume_chunks, job_text, model)
            if score >= SIMILARITY_THRESHOLD:
                matched_jobs.append({
                    "Title": job["title"],
                    "Company": job["company"],
                    "Location": job["location"],
                    "Score": score,
                    "Apply Link": job["link"]
                })

        matched_jobs = sorted(matched_jobs, key=lambda x: x["Score"], reverse=True)

        if matched_jobs:
            st.success(f"✅ {len(matched_jobs)} jobs matched above {SIMILARITY_THRESHOLD}% threshold.")
            df = pd.DataFrame(matched_jobs)
            st.dataframe(df)

            # Optional download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results", data=csv, file_name="matched_jobs.csv", mime="text/csv")
        else:
            st.warning("❌ No jobs matched your resume above the threshold.")
