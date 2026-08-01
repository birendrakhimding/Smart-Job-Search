
# Streamlit Web Application — Smart Job Search


import streamlit as st
import os
import re
import json
import requests
import numpy as np
from datetime import datetime

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Job Search",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load API Keys (from Streamlit secrets or .env)
# ─────────────────────────────────────────────────────────────────────────────

def load_keys():
    """Load API keys from Streamlit secrets (deployed) or .env (local)."""
    try:
        # Streamlit Cloud deployment
        return {
            "gemini": st.secrets["GEMINI_API_KEY"],
            "jsearch": st.secrets["JSEARCH_API_KEY"],
            "adzuna_id": st.secrets["ADZUNA_APP_ID"],
            "adzuna_key": st.secrets["ADZUNA_APP_KEY"],
        }
    except:
        # Local development with .env
        from dotenv import load_dotenv
        load_dotenv()
        return {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "jsearch": os.getenv("JSEARCH_API_KEY"),
            "adzuna_id": os.getenv("ADZUNA_APP_ID"),
            "adzuna_key": os.getenv("ADZUNA_APP_KEY"),
        }

keys = load_keys()

# ─────────────────────────────────────────────────────────────────────────────
# Load Models (cached so they only load once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_gemini():
    genai.configure(api_key=keys["gemini"])
    return genai.GenerativeModel("gemini-3.5-flash-lite")

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("fine_tuned_minilm")

gemini_model = load_gemini()
st_model = load_sentence_transformer()

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file."""
    import fitz
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def clean_text_for_model(text):
    """Light cleaning matching the fine-tuning preprocessing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(text.split()[:256])


def parse_resume(resume_text):
    """Use Gemini to extract structured information from the resume."""
    prompt = f"""Analyze this resume and extract the following information. 
Respond ONLY in valid JSON format with no markdown backticks or extra text.

{{
    "name": "candidate's full name",
    "email": "email if found, otherwise null",
    "phone": "phone if found, otherwise null",
    "skills": ["list", "of", "key", "skills"],
    "experience_years": estimated total years of experience as a number,
    "job_titles": ["list", "of", "previous", "job", "titles"],
    "education": ["list", "of", "degrees", "or", "certifications"],
    "category": "best matching category from this list: ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER",
    "summary": "2-3 sentence professional summary of the candidate"
}}

Resume:
{resume_text[:3000]}
"""
    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Error parsing resume: {e}")
        return None


def generate_search_queries(parsed):
    """Use Gemini to generate job search queries."""
    prompt = f"""Based on this candidate profile, generate 3-5 job search queries 
that would find the best matching job postings. Each query should be 3-6 words.
Respond ONLY as a JSON array of strings, no markdown.

Candidate Profile:
- Skills: {', '.join(parsed.get('skills', [])[:10])}
- Job Titles: {', '.join(parsed.get('job_titles', [])[:5])}
- Category: {parsed.get('category', '')}
- Experience: {parsed.get('experience_years', 'N/A')} years

Example output: ["software engineer python", "backend developer", "full stack developer"]
"""
    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        queries = json.loads(response_text)
        return queries if queries else [parsed.get("category", "developer")]
    except:
        return parsed.get("job_titles", ["developer"])[:3]


def search_jsearch(query, num_results=5):
    """Fetch jobs from JSearch API."""
    url = "https://jsearch.p.rapidapi.com/search-v2"
    headers = {
        "x-rapidapi-key": keys["jsearch"],
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
        "Content-Type": "application/json"
    }
    params = {"query": query, "page": "1", "num_pages": "1", "date_posted": "month"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        data = response.json()
        jobs = []
        for job in data.get("data", {}).get("jobs", [])[:num_results]:
            jobs.append({
                "title": job.get("job_title", "N/A"),
                "company": job.get("employer_name", "N/A"),
                "location": job.get("job_city", "Remote"),
                "description": job.get("job_description", "")[:500],
                "full_description": job.get("job_description", ""),
                "salary_min": job.get("job_min_salary"),
                "salary_max": job.get("job_max_salary"),
                "apply_link": job.get("job_apply_link", ""),
                "source": "JSearch",
            })
        return jobs
    except:
        return []


def search_adzuna(query, num_results=5):
    """Fetch jobs from Adzuna API."""
    url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
    params = {
        "app_id": keys["adzuna_id"],
        "app_key": keys["adzuna_key"],
        "results_per_page": num_results,
        "what": query,
        "max_days_old": 30,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        jobs = []
        for job in data.get("results", [])[:num_results]:
            jobs.append({
                "title": job.get("title", "N/A"),
                "company": job.get("company", {}).get("display_name", "N/A"),
                "location": job.get("location", {}).get("display_name", "Remote"),
                "description": job.get("description", "")[:500],
                "full_description": job.get("description", ""),
                "salary_min": job.get("salary_min"),
                "salary_max": job.get("salary_max"),
                "apply_link": job.get("redirect_url", ""),
                "source": "Adzuna",
            })
        return jobs
    except:
        return []


def search_all_jobs(queries):
    """Search both APIs with all queries, deduplicate results."""
    all_jobs = []
    seen = set()

    for query in queries:
        for job in search_jsearch(query) + search_adzuna(query):
            key = f"{job['title'].lower().strip()}_{job['company'].lower().strip()}"
            if key not in seen:
                seen.add(key)
                all_jobs.append(job)

    return all_jobs


def score_jobs(resume_text, jobs):
    """Score jobs using the fine-tuned Sentence Transformer."""
    if not jobs:
        return []

    clean_resume = clean_text_for_model(resume_text)
    resume_embedding = st_model.encode([clean_resume])

    job_texts = [clean_text_for_model(j.get("full_description", j.get("description", ""))) for j in jobs]
    job_embeddings = st_model.encode(job_texts, show_progress_bar=False)

    similarities = cos_sim(resume_embedding, job_embeddings)[0]

    scored = []
    for i, job in enumerate(jobs):
        job_copy = job.copy()
        job_copy["similarity_score"] = round(float(similarities[i]), 4)
        scored.append(job_copy)

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:10]





# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.title("🔍 Smart Job Search")
st.markdown("*Upload your resume and find the best matching jobs using AI-powered semantic matching.*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **How it works:**
    1. Upload your resume (PDF or text)
    2. AI parses your skills and experience
    3. Jobs are fetched from JSearch & Adzuna
    4. Fine-tuned deep learning model scores matches
    5. Get your top 10 matching jobs

    **Built with:**
    - Gemini 3.5 Flash-Lite
    - Fine-tuned all-MiniLM-L6-v2
    - LangGraph Pipeline
    - JSearch & Adzuna APIs
    """)

    st.markdown("---")
    st.markdown("**Capstone Project**")
    st.markdown("Birendra Khimding")
    st.markdown("MS Applied AI")
    st.markdown("University of San Diego")

# File Upload
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "txt"],
        help="Supported formats: PDF, TXT"
    )

with col2:
    st.markdown("")
    st.markdown("")
    use_sample = st.checkbox("Use sample resume instead")

# Sample resume
sample_resume = """John Smith
john.smith@email.com | (555) 123-4567 | San Diego, CA

PROFESSIONAL SUMMARY
Experienced software engineer with 5 years of experience in full-stack development.
Proficient in Python, JavaScript, React, and cloud technologies.

EXPERIENCE
Senior Software Engineer | TechCorp Inc. | 2022 - Present
- Developed microservices using Python and FastAPI
- Built front-end applications with React and TypeScript
- Deployed applications on AWS using Docker and Kubernetes

Software Developer | WebSolutions LLC | 2020 - 2022
- Built full-stack web applications using Django and React
- Designed and optimized PostgreSQL databases

EDUCATION
Master of Science in Computer Science | University of San Diego | 2019

SKILLS
Python, JavaScript, TypeScript, React, Django, Flask, FastAPI, PostgreSQL,
MongoDB, AWS, Docker, Kubernetes, Git, CI/CD, REST APIs"""

# Process Resume
resume_text = None

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8")
elif use_sample:
    resume_text = sample_resume

if resume_text:
    # Show resume preview
    with st.expander("📄 Resume Preview", expanded=False):
        st.text(resume_text[:2000])

    # Run Pipeline
    if st.button("🚀 Find Matching Jobs", type="primary", use_container_width=True):

        # Step 1: Parse Resume
        with st.status("Running AI Pipeline...", expanded=True) as status:

            st.write("📄 Parsing resume with Gemini...")
            parsed = parse_resume(resume_text)

            if not parsed:
                st.error("Failed to parse resume. Please try again.")
                st.stop()

            # Show parsed info
            st.write(f"✅ **{parsed.get('name', 'N/A')}** | {parsed.get('category', 'N/A')} | {parsed.get('experience_years', 'N/A')} years experience")

            # Step 2: Generate Queries
            st.write("🔍 Generating search queries...")
            queries = generate_search_queries(parsed)
            st.write(f"✅ Queries: {', '.join(queries)}")

            # Step 3: Search Jobs
            st.write("🌐 Searching JSearch & Adzuna...")
            jobs = search_all_jobs(queries)
            st.write(f"✅ Found {len(jobs)} unique jobs")

            if not jobs:
                st.error("No jobs found. Try uploading a different resume.")
                st.stop()

            # Step 4: Score Matches
            st.write("🎯 Scoring matches with fine-tuned model...")
            scored_jobs = score_jobs(resume_text, jobs)
            st.write(f"✅ Top {len(scored_jobs)} matches ranked")

            status.update(label="Pipeline complete!", state="complete")

        # Store results in session state
        st.session_state["parsed"] = parsed
        st.session_state["scored_jobs"] = scored_jobs
        st.session_state["queries"] = queries

    # Display Results
    if "scored_jobs" in st.session_state:
        parsed = st.session_state["parsed"]
        scored_jobs = st.session_state["scored_jobs"]

        st.markdown("---")

        # Candidate Summary
        st.subheader("👤 Candidate Profile")
        col1, col2, col3 = st.columns(3)
        col1.metric("Name", parsed.get("name", "N/A"))
        col2.metric("Category", parsed.get("category", "N/A"))
        col3.metric("Experience", f"{parsed.get('experience_years', 'N/A')} years")

        skills = parsed.get("skills", [])
        if skills:
            st.markdown("**Skills:** " + " • ".join(skills[:10]))

        st.markdown("---")

        # Job Results
        st.subheader(f"🏆 Top {len(scored_jobs)} Job Matches")

        for i, job in enumerate(scored_jobs):
            score = job.get("similarity_score", 0)

            # Color based on score
            if score >= 0.6:
                score_color = "🟢"
            elif score >= 0.4:
                score_color = "🟡"
            else:
                score_color = "🔴"

            # Salary formatting
            if job.get("salary_min") and job.get("salary_max"):
                salary = f"${job['salary_min']:,.0f} - ${job['salary_max']:,.0f}"
            elif job.get("salary_min"):
                salary = f"${job['salary_min']:,.0f}+"
            else:
                salary = "Not listed"

            # Job card
            with st.container():
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"### {i+1}. {job['title']}")
                    st.markdown(f"**{job['company']}** | 📍 {job.get('location', 'N/A')} | 💰 {salary} | 📡 {job.get('source', 'N/A')}")

                with col2:
                    st.markdown(f"### {score_color} {score:.1%}")

                # Expandable details
                with st.expander("View Details"):
                    st.markdown(f"**Description:** {job.get('description', 'N/A')}")

                    if job.get("apply_link"):
                        st.markdown(f"[🔗 Apply Here]({job['apply_link']})")

                st.markdown("---")
else:
    # Landing state
    st.info("👆 Upload your resume or check 'Use sample resume' to get started.")
