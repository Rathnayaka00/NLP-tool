import sys
import os
import shutil
import asyncio
import warnings
from pathlib import Path
from typing import Union, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except Exception:
    pass

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from preprocess.extract_text import parse_pdf_to_markdown
from preprocess.pre_process import preprocess_text_for_summarization, download_nltk_data
from summarization.summery import summarize_text
from Translation.english_to_sinhala import translate_en_to_si

app = FastAPI(title="NLP Tool API", version="1.0.0")

# Enable CORS for local UI development and general access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("Server starting up...")
    download_nltk_data()

@app.get("/")
def read_root():
    return {"message": "Welcome to the NLP Processing API!"}


@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...), summary_sentences: Optional[int] = None):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_path = uploads_dir / Path(file.filename).name

    try:
        with saved_path.open("wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
    finally:
        await file.close()
    try:
        print(f"Parsing PDF: {saved_path}")
        markdown_text = await run_in_threadpool(parse_pdf_to_markdown, str(saved_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")

    print("Pre-processing the extracted text...")
    cleaned_text = preprocess_text_for_summarization(markdown_text)

    try:
        # Compute dynamic summary length from cleaned text if not provided
        if summary_sentences is None:
            word_count = len(cleaned_text.split())
            if word_count < 120:
                n_sentences = 3
            elif word_count < 250:
                n_sentences = 5
            elif word_count < 500:
                n_sentences = 7
            elif word_count < 1000:
                n_sentences = 9
            else:
                n_sentences = 12
        else:
            n_sentences = max(3, min(12, int(summary_sentences)))

        # Use the original extracted text for summarization to retain punctuation and structure
        summary = await run_in_threadpool(summarize_text, markdown_text, n_sentences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize text: {e}")
    
    try:
        print("Translating summary to Sinhala...")
        sinhala_summary = await run_in_threadpool(translate_en_to_si, summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to translate text: {e}")

    print("\nFinal Cleaned Text")
    print(cleaned_text)

    print("\nSummary")
    print(summary)

    print("\nSinhala Summary")
    print(sinhala_summary)

    
    return {
        "status": "ok",
        "filename": file.filename,
        "original_chars": len(markdown_text),
        "cleaned_chars": len(cleaned_text),
        "cleaned_text_preview": cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
        "summary": summary,
        "sinhala_summary": sinhala_summary
    }