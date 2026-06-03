from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pdfplumber
import uvicorn
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

# -----------------------------------
# LOAD ENV
# -----------------------------------

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

AZURE_OPENAI_SUMMARY_MODEL = os.getenv("AZURE_OPENAI_SUMMARY_MODEL")

AZURE_OPENAI_SUMMARY_API_VERSION = os.getenv(
    "AZURE_OPENAI_SUMMARY_API_VERSION"
)

if not AZURE_OPENAI_ENDPOINT:
    raise ValueError("AZURE_OPENAI_ENDPOINT not found")

if not AZURE_OPENAI_API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY not found")

client = AzureOpenAI(

    azure_endpoint=AZURE_OPENAI_ENDPOINT,

    api_key=AZURE_OPENAI_API_KEY,

    api_version=AZURE_OPENAI_SUMMARY_API_VERSION,
)

# -----------------------------------
# APP
# -----------------------------------

app = FastAPI()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------------
# PYDANTIC MODELS
# -----------------------------------

class Education(BaseModel):

    degree: Optional[str] = None

    institution: Optional[str] = None

    year: Optional[str] = None

    cgpa: Optional[str] = None


class Experience(BaseModel):

    title: Optional[str] = None

    company: Optional[str] = None

    duration: Optional[str] = None

    description: Optional[str] = None


class Project(BaseModel):

    name: Optional[str] = None

    description: Optional[str] = None


class ResumeData(BaseModel):

    name: Optional[str] = None

    email: Optional[str] = None

    phone: Optional[str] = None

    linkedin: Optional[str] = None

    github: Optional[str] = None

    skills: List[str] = []

    education: List[Education] = []

    experience: List[Experience] = []

    certifications: List[str] = []

    projects: List[Project] = []

    languages: List[str] = []

    summary: Optional[str] = None


# -----------------------------------
# PDF TEXT EXTRACTION
# -----------------------------------

def extract_text_from_pdf(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text


# -----------------------------------
# LLM EXTRACTION
# -----------------------------------

def extract_resume_with_llm(text):

    prompt = f"""
    Extract all important information from the resume.

    Return ONLY valid JSON.

    Required JSON format:

    {{
        "name": "",
        "email": "",
        "phone": "",
        "linkedin": "",
        "github": "",
        "skills": [],

        "education": [
            {{
                "degree": "",
                "institution": "",
                "year": "",
                "cgpa": ""
            }}
        ],

        "experience": [
            {{
                "title": "",
                "company": "",
                "duration": "",
                "description": ""
            }}
        ],

        "certifications": [],

        "projects": [
            {{
                "name": "",
                "description": ""
            }}
        ],

        "languages": [],

        "summary": ""
    }}

    Resume Text:
    {text}
    """

    response = client.chat.completions.create(

        model=AZURE_OPENAI_SUMMARY_MODEL,

        messages=[

            {
                "role": "system",
                "content": (
                    "You are an expert resume parser. "
                    "Always return ONLY valid JSON."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0,

        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    return json.loads(content)


# -----------------------------------
# API ROUTE
# -----------------------------------

@app.post("/parse-resume")

async def parse_resume(
    file: UploadFile = File(...)
):

    try:

        # SAVE PDF

        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        with open(file_path, "wb") as buffer:

            buffer.write(await file.read())

        # EXTRACT PDF TEXT

        text = extract_text_from_pdf(file_path)

        if not text.strip():

            return JSONResponse(

                status_code=400,

                content={
                    "error": "No text found in PDF"
                }
            )

        # SEND TO LLM

        extracted_data = extract_resume_with_llm(text)

        # VALIDATE WITH PYDANTIC

        resume = ResumeData(**extracted_data)

        return JSONResponse(
            content=resume.model_dump()
        )

    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={
                "error": str(e)
            }
        )


# -----------------------------------
# RUN APP
# -----------------------------------

if __name__ == "__main__":

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
