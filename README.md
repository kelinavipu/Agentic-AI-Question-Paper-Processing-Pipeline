# Extracta — AI Question Paper Processing Pipeline

> An agentic AI system that ingests university exam PDFs, extracts and structures all questions via OCR + LLM, generates academic model answer sheets, and exports everything as Excel + PDF.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square&logo=fastapi)
![Groq](https://img.shields.io/badge/Groq-Llama--3.3--70b-orange?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Features

- **PDF Ingestion** — Renders every page at 200 DPI using PyMuPDF
- **OCR Pipeline** — Tesseract OCR with adaptive preprocessing for clean text extraction
- **LLM Cleaning** — Groq Llama-3.3-70b fixes OCR noise, reconstructs split equations
- **Hierarchical Parsing** — Extracts full Q → Sub-Q → Sub-Sub-Q tree (handles "write short note on any N" patterns)
- **Diagram Detection** — OpenCV contour detection crops circuit diagrams, tables, and flowcharts precisely
- **Curiosity Sparks** — AI-generated Real-World Connection, Mind-Blowing Fact, and Curious Quest for each question
- **Model Answer Sheet** — Parallel 6-key Groq pool solves all questions categorized as Analysis / Define / Justify / Differences / General
- **PDF Export** — University-style formatted answer PDF with section headings, tables, and marks-aware point counts
- **Excel Export** — Structured workbook with all question hierarchy levels
- **Smart Key Rotation** — Detects daily TPD exhaustion vs per-minute rate limits and rotates keys intelligently

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Agent Orchestration | LangGraph (StateGraph) |
| LLM | Groq — Llama-3.3-70b-Versatile |
| OCR | Tesseract via pytesseract |
| PDF Rendering | PyMuPDF (fitz) |
| Image Processing | OpenCV + NumPy |
| PDF Generation | fpdf2 |
| Excel Generation | openpyxl |
| Frontend | Vanilla HTML/CSS/JS + marked.js + MathJax |

---

## Project Structure

```
Extracta/
├── agent.py          # Full agentic pipeline (LangGraph nodes + PDF compiler)
├── main.py           # FastAPI server + all API endpoints
├── static/
│   ├── index.html    # Frontend UI
│   ├── app.js        # Frontend logic (rendering, polling, sparks, answers)
│   └── style.css     # Dark-mode premium UI styles
├── output/           # Generated files (Excel, PDFs, cropped images)
├── .env              # API keys (NOT committed — see below)
├── .env.example      # Template for setting up your own keys
└── requirements.txt  # Python dependencies
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kelinavipu/Agentic_AI_Question-Paper-Processing-Pipeline.git
cd Agentic_AI_Question-Paper-Processing-Pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

- **Windows**: Download from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`

### 4. Configure API Keys

Copy the example env file and fill in your [Groq API keys](https://console.groq.com/):

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_SUMMARY_KEY=gsk_your_summary_key_here

GROQ_EXT1=gsk_your_ext_key_1
GROQ_EXT2=gsk_your_ext_key_2
GROQ_EXT3=gsk_your_ext_key_3
GROQ_EXT4=gsk_your_ext_key_4
GROQ_EXT5=gsk_your_ext_key_5
GROQ_EXT6=gsk_your_ext_key_6
```

### 5. Run

```bash
python main.py
```

Open **http://localhost:8000** in your browser.

---

## Usage

1. **Upload** a university exam PDF (Mumbai University, generic, etc.)
2. **Select** your university format
3. Watch the **live progress pipeline** extract all questions
4. **Click any sub-question** to see the AI curiosity overview panel
5. Click **Generate AI Model Answer Sheet** to run the parallel 6-key Groq agent
6. **Download** the formatted PDF answer sheet or Excel workbook

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/extract` | Upload PDF and start pipeline |
| `GET` | `/api/status/{task_id}` | Poll pipeline progress |
| `GET` | `/api/result/{task_id}` | Get structured question tree |
| `POST` | `/api/spark` | Generate curiosity spark for a question |
| `POST` | `/api/answers/{task_id}` | Start parallel model answer generation |
| `GET` | `/api/answers/status/{task_id}` | Poll answer generation progress |
| `GET` | `/api/download/{task_id}` | Download Excel workbook |
| `GET` | `/api/download-answers/{task_id}` | Download model answer PDF |
| `GET` | `/api/media/{task_id}/{filename}` | Serve cropped diagram images |

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `GROQ_SUMMARY_KEY` | Used for spark generation, OCR cleaning, header extraction |
| `GROQ_EXT1`–`GROQ_EXT6` | Rotated pool for question extraction and answer generation |

---

## License

MIT — free to use, modify, and distribute.
