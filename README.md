# 🔍 Smart Job Search Using Deep Learning

A semantic resume-to-job matching system that uses deep learning to find the most relevant job postings for your resume. Built as a capstone project for MS Applied AI at the University of San Diego.

**[Try the Live App →](https://smartjobsearch.streamlit.app)**
> **Note:** The app is hosted on Streamlit Community Cloud (free tier) and goes to sleep after 12 hours of inactivity. If you see a sleeping page, click "Yes, get this app back up!" and wait about 1–2 minutes for it to reload.

## How It Works

Upload a resume (PDF or text) and the app runs a 4-node agentic pipeline that:

1. **Parses your resume** using Gemini 3.5 Flash-Lite — extracts skills, experience, and job category
2. **Generates search queries** tailored to your profile using Gemini
3. **Fetches live job postings** from JSearch and Adzuna APIs
4. **Scores and ranks jobs** using a fine-tuned Sentence Transformer (all-MiniLM-L6-v2) by cosine similarity

Returns the top 10 matched jobs with similarity scores, salary info, and apply links in 15–30 seconds.

## Project Structure

| File | Description |
|------|-------------|
| `eda.ipynb` | Exploratory data analysis on resume and job datasets |
| `basemodel.ipynb` | Siamese Bidirectional LSTM baseline model (65.1% accuracy) |
| `SentenceTransformer.ipynb` | Fine-tuned all-MiniLM-L6-v2 model (83.9% accuracy) |
| `agent.ipynb` | LangGraph agentic pipeline (4 nodes) |
| `app.py` | Streamlit web application |
| `requirements.txt` | Python dependencies |
| `.python-version` | Pins Python 3.11 for Streamlit Cloud |

## Model Comparison

| Model | Accuracy | MSE | F1 (Weighted) |
|-------|----------|-----|----------------|
| Siamese LSTM (from scratch) | 65.1% | 0.0936 | 0.65 |
| Pre-trained MiniLM (no fine-tuning) | 33.2% | — | — |
| **Fine-tuned MiniLM** | **83.9%** | **0.0669** | **0.84** |

The fine-tuned Sentence Transformer outperformed the LSTM baseline by **+18.8 percentage points**, confirming that transfer learning with domain-specific fine-tuning is significantly more effective than training from scratch.

## Datasets

- **LiveCareer Resume Dataset** — 2,484 resumes across 24 job categories [Kaggle Link](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **LinkedIn Job Postings** — 101,492 filtered postings across 24 matching categories [Kaggle Link](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

## Tech Stack

- **Models:** PyTorch, Sentence Transformers, Hugging Face
- **Pipeline:** LangGraph, LangChain, Google Gemini API
- **Job APIs:** JSearch (RapidAPI), Adzuna
- **Web App:** Streamlit
- **Other:** scikit-learn, PyMuPDF, NumPy

## Run Locally

```bash
# Clone the repo
git clone https://github.com/birendrakhimding/Smart-Job-Search.git
cd Smart-Job-Search

# Install dependencies
pip install -r requirements.txt

# Add API keys to a .env file
GEMINI_API_KEY="your-key"
JSEARCH_API_KEY="your-key"
ADZUNA_APP_ID="your-id"
ADZUNA_APP_KEY="your-key"

# Run the app
streamlit run app.py
```

