# Streamlit Web Application — Smart Job Search (with LangGraph)


import streamlit as st
import os
import re
import json
import requests
import numpy as np
from typing import TypedDict, List
from datetime import datetime

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from langgraph.graph import StateGraph, END

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
    try:
        return {
            "gemini": st.secrets["GEMINI_API_KEY"],
            "jsearch": st.secrets["JSEARCH_API_KEY"],
            "adzuna_id": st.secrets["ADZUNA_APP_ID"],
            "adzuna_key": st.secrets["ADZUNA_APP_KEY"],
        }
    except:
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
    return SentenceTransformer("minsolimbu/smart-job-search-model")

gemini_model = load_gemini()
st_model = load_sentence_transformer()

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file):
    import fitz
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def clean_text_for_model(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(text.split()[:256])

# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Agent State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    resume_text: str
    parsed_resume: dict
    search_queries: List[str]
    jobs: List[dict]
    scored_jobs: List[dict]
    status: str
    errors: List[str]

# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Node Functions
# ─────────────────────────────────────────────────────────────────────────────

def parse_resume_node(state: AgentState) -> AgentState:
    resume_text = state["resume_text"]
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
        state["parsed_resume"] = json.loads(response_text)
        state["status"] = "resume_parsed"
    except Exception as e:
        state["errors"].append(f"Resume parsing error: {e}")
        state["parsed_resume"] = {
            "skills": [], "category": "INFORMATION-TECHNOLOGY",
            "summary": resume_text[:200], "name": "Unknown"
        }
        state["status"] = "resume_parse_failed"
    return state


def generate_queries_node(state: AgentState) -> AgentState:
    parsed = state["parsed_resume"]
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
        state["search_queries"] = queries if queries else [parsed.get("category", "developer")]
        state["status"] = "queries_generated"
    except Exception as e:
        state["errors"].append(f"Query generation error: {e}")
        state["search_queries"] = parsed.get("job_titles", ["developer"])[:3]
        state["status"] = "queries_fallback"
    return state


def search_jobs_node(state: AgentState) -> AgentState:
    queries = state["search_queries"]
    all_jobs = []
    seen = set()

    for query in queries:
        # JSearch
        try:
            url = "https://jsearch.p.rapidapi.com/search-v2"
            headers = {
                "x-rapidapi-key": keys["jsearch"],
                "x-rapidapi-host": "jsearch.p.rapidapi.com",
                "Content-Type": "application/json"
            }
            params = {"query": query, "page": "1", "num_pages": "1", "date_posted": "month"}
            response = requests.get(url, headers=headers, params=params, timeout=15)
            data = response.json()
            for job in data.get("data", {}).get("jobs", [])[:5]:
                j = {
                    "title": job.get("job_title", "N/A"),
                    "company": job.get("employer_name", "N/A"),
                    "location": job.get("job_city", "Remote"),
                    "description": job.get("job_description", "")[:500],
                    "full_description": job.get("job_description", ""),
                    "salary_min": job.get("job_min_salary"),
                    "salary_max": job.get("job_max_salary"),
                    "apply_link": job.get("job_apply_link", ""),
                    "source": "JSearch",
                }
                key = f"{j['title'].lower().strip()}_{j['company'].lower().strip()}"
                if key not in seen:
                    seen.add(key)
                    all_jobs.append(j)
        except:
            pass

        # Adzuna
        try:
            url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
            params = {
                "app_id": keys["adzuna_id"],
                "app_key": keys["adzuna_key"],
                "results_per_page": 5,
                "what": query,
                "max_days_old": 30,
            }
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            for job in data.get("results", [])[:5]:
                j = {
                    "title": job.get("title", "N/A"),
                    "company": job.get("company", {}).get("display_name", "N/A"),
                    "location": job.get("location", {}).get("display_name", "Remote"),
                    "description": job.get("description", "")[:500],
                    "full_description": job.get("description", ""),
                    "salary_min": job.get("salary_min"),
                    "salary_max": job.get("salary_max"),
                    "apply_link": job.get("redirect_url", ""),
                    "source": "Adzuna",
                }
                key = f"{j['title'].lower().strip()}_{j['company'].lower().strip()}"
                if key not in seen:
                    seen.add(key)
                    all_jobs.append(j)
        except:
            pass

    state["jobs"] = all_jobs
    state["status"] = "jobs_found" if all_jobs else "no_jobs_found"
    return state


def score_matches_node(state: AgentState) -> AgentState:
    resume_text = state["resume_text"]
    jobs = state["jobs"]

    if not jobs:
        state["scored_jobs"] = []
        state["status"] = "no_jobs_to_score"
        return state

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
    state["scored_jobs"] = scored[:10]
    state["status"] = "jobs_scored"
    return state

# ─────────────────────────────────────────────────────────────────────────────
# Build LangGraph Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def build_pipeline():
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_resume", parse_resume_node)
    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("search_jobs", search_jobs_node)
    workflow.add_node("score_matches", score_matches_node)

    workflow.set_entry_point("parse_resume")
    workflow.add_edge("parse_resume", "generate_queries")
    workflow.add_edge("generate_queries", "search_jobs")
    workflow.add_edge("search_jobs", "score_matches")
    workflow.add_edge("score_matches", END)

    return workflow.compile()

pipeline = build_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔍 Smart Job Search")
st.markdown("*Upload your resume and find the best matching jobs using AI-powered semantic matching.*")
st.markdown("---")

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

resume_text = None

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8")
elif use_sample:
    resume_text = sample_resume

if resume_text:
    with st.expander("📄 Resume Preview", expanded=False):
        st.text(resume_text[:2000])

    if st.button("🚀 Find Matching Jobs", type="primary", use_container_width=True):

        with st.status("Running LangGraph Pipeline...", expanded=True) as status:

            state = {
                "resume_text": resume_text,
                "parsed_resume": {},
                "search_queries": [],
                "jobs": [],
                "scored_jobs": [],
                "status": "started",
                "errors": [],
            }

            st.write("📄 Node 1: Parsing resume with Gemini...")
            state = parse_resume_node(state)
            parsed = state["parsed_resume"]
            st.write(f"✅ **{parsed.get('name', 'N/A')}** | {parsed.get('category', 'N/A')} | {parsed.get('experience_years', 'N/A')} years experience")

            st.write("🔍 Node 2: Generating search queries...")
            state = generate_queries_node(state)
            queries = state["search_queries"]
            st.write(f"✅ Queries: {', '.join(queries)}")

            st.write("🌐 Node 3: Searching JSearch & Adzuna...")
            state = search_jobs_node(state)
            st.write(f"✅ Found {len(state['jobs'])} unique jobs")

            st.write("🎯 Node 4: Scoring matches with fine-tuned model...")
            state = score_matches_node(state)
            scored_jobs = state["scored_jobs"]
            st.write(f"✅ Top {len(scored_jobs)} matches ranked")

            if state.get("errors"):
                for err in state["errors"]:
                    st.warning(err)

            status.update(label="Pipeline complete!", state="complete")

        st.session_state["parsed"] = parsed
        st.session_state["scored_jobs"] = scored_jobs
        st.session_state["queries"] = queries

    if "scored_jobs" in st.session_state:
        parsed = st.session_state["parsed"]
        scored_jobs = st.session_state["scored_jobs"]

        st.markdown("---")

        st.subheader("👤 Candidate Profile")
        col1, col2, col3 = st.columns(3)
        col1.metric("Name", parsed.get("name", "N/A"))
        col2.metric("Category", parsed.get("category", "N/A"))
        col3.metric("Experience", f"{parsed.get('experience_years', 'N/A')} years")

        skills = parsed.get("skills", [])
        if skills:
            st.markdown("**Skills:** " + " • ".join(skills[:10]))

        st.markdown("---")

        st.subheader(f"🏆 Top {len(scored_jobs)} Job Matches")

        for i, job in enumerate(scored_jobs):
            score = job.get("similarity_score", 0)

            if score >= 0.6:
                score_color = "🟢"
            elif score >= 0.4:
                score_color = "🟡"
            else:
                score_color = "🔴"

            if job.get("salary_min") and job.get("salary_max"):
                salary = f"${job['salary_min']:,.0f} - ${job['salary_max']:,.0f}"
            elif job.get("salary_min"):
                salary = f"${job['salary_min']:,.0f}+"
            else:
                salary = "Not listed"

            with st.container():
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"### {i+1}. {job['title']}")
                    st.markdown(f"**{job['company']}** | 📍 {job.get('location', 'N/A')} | 💰 {salary} | 📡 {job.get('source', 'N/A')}")

                with col2:
                    st.markdown(f"### {score_color} {score:.1%}")

                with st.expander("View Details"):
                    st.markdown(f"**Description:** {job.get('description', 'N/A')}")

                    if job.get("apply_link"):
                        st.markdown(f"[🔗 Apply Here]({job['apply_link']})")

                st.markdown("---")
else:
    st.info("👆 Upload your resume or check 'Use sample resume' to get started.")