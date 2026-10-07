# ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and suggests concrete improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as PDF, DOCX or TXT
- Optional job description for keyword matching
- Overall ATS score plus breakdown (keywords, formatting, content, completeness)
- Strengths, missing keywords, prioritized improvements and before/after bullet rewrites
- Downloadable JSON report

## Run locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```
Get a free key at https://aistudio.google.com/apikey. You can also paste the key in the app sidebar.

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt` and `README.md` to a GitHub repository.
2. Go to https://share.streamlit.io and click **Create app**.
3. Choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings > Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Notes
- The default model is `gemini-3.5-flash`; if it fails, the app falls back to `gemini-2.5-flash`. You can change the model name in the sidebar.
- Scanned (image-only) PDFs can't be read; use a text-based PDF or DOCX.
- Never commit your API key to GitHub.
