#  YouTube Video Transcript Summarizer + similar video suggestions

A production-ready YouTube video analysis system built using **LangChain, LangGraph, FastAPI, Groq LLM, and structured Pydantic validation**.

This project extracts a YouTube transcript and performs structured AI analysis in a **single optimized LLM call** to minimize API usage and avoid rate limits.

---

## 🚀 What This Project Does

Given a YouTube video URL, the system:


Extracts the transcript of any YouTube video.
Summarizes the content into human-readable text.
Identifies key topics and keywords.
Suggests related videos for deeper exploration.
Generates thoughtful questions for self-checks.
Provides actionable next steps for continued learning.
In short, it’s not just a summarizer—it’s your personal mini learning assistant.

---

## 🧠 Architecture Overview

### 🔹 LangGraph Pipeline

<img width="1120" height="717" alt="Screenshot 2026-02-22 221647" src="https://github.com/user-attachments/assets/f627f2a9-1bf9-4191-9297-2458ab86e1e3" />

We use:

* **Groq Llama 3.1 8B Instant**
* Structured output via Pydantic schema
* Only **ONE LLM call per request**
* Reduced token overhead
* Rate-limit optimized

---

## 🏗 Tech Stack

* Python
* FastAPI
* LangGraph
* LangChain Core
* Groq LLM (Llama 3.1 8B Instant)
* Pydantic (Strict Validation)
* YouTube Transcript API
* Uvicorn

---
## 📦 Project Structure

```
Youtube_Video_Summarizer/
│
├── app.py              # FastAPI app
├── main.py             # LangGraph pipeline
├── langgraph.json
├── requirements.txt
└── .gitignore
└── README.md

```
---

## 🔐 Environment Setup

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=project name
```

---

## ⚙ Installation

```bash
git clone <repo-url>
cd Youtube_Video_Summarizer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶ Running the API

```bash
uvicorn app:app --reload
```

API will run at:

```
http://127.0.0.1:8000
```

---

## 📌 API Endpoints

### GET `/`

Health check

Response:

```json
{
  "Hello": "world"
}
```

---

### POST `/summarizer`

Request:

```
/summarizer?video_url=<youtube_url>
```

Example:

```
http://127.0.0.1:8000/summarizer?video_url=https://www.youtube.com/watch?v=XXXXXXXXXXX
```

Response:

```json
{
  "video_url": "...",
  "video_id": "...",
  "summary": "...",
  "keywords": ["AI", "Neural Networks"],
  "questions": [...],
  "next_steps": [...],
  "video_suggestions": [...]
}
```

