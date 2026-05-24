import os
import re
import cv2
import json
import numpy as np
import pytesseract
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pathlib import Path

import fitz  # PyMuPDF
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage

import time
import threading
import concurrent.futures
from fpdf import FPDF

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

# Load .env from project root
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path)
    print(f"[Extracta Agent] Loaded .env from {_env_path}")
except ImportError:
    pass  # python-dotenv not installed, fall back to system env

# --------------------------------------------------------------------------
# CONFIG & TESSERACT PATH RESOLUTION
# --------------------------------------------------------------------------

def _find_tesseract() -> str:
    candidates = [
        r"C:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\tesseract.exe",
        r"C:\Users\Kelina\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[Extracta Agent] Found Tesseract at: {p}")
            return p
    print("[Extracta Agent] Tesseract not found in standard paths, using PATH fallback.")
    return "tesseract"

pytesseract.pytesseract.tesseract_cmd = _find_tesseract()

# Summary / Spark LLM key (GROQ_SUMMARY_KEY in .env)
GROQ_SUMMARY_KEY = (
    os.environ.get("GROQ_SUMMARY_KEY")
    or os.environ.get("GROQ_API_KEY", "")
)

try:
    LLM = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_SUMMARY_KEY)
    LLM_AVAILABLE = True
    print("[Extracta Agent] Groq LLM (summarykey) initialized successfully.")
except Exception as e:
    LLM_AVAILABLE = False
    print(f"[Extracta Agent] WARNING: Groq LLM initialization failed: {e}")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OCR_CONFIG  = r"--oem 3 --psm 6"
MAX_RETRIES = 3

task_progress_logs: Dict[str, List[Dict]] = {}

def update_task_progress(task_id: str, node_name: str, percentage: int, status_message: str):
    if task_id not in task_progress_logs:
        task_progress_logs[task_id] = []
    task_progress_logs[task_id].append({
        "node": node_name,
        "progress": percentage,
        "message": status_message
    })
    print(f"[{task_id}] Node: {node_name} | Progress: {percentage}% | {status_message}")


# --------------------------------------------------------------------------
# SHARED STATE SCHEMA
# --------------------------------------------------------------------------

class QPState(TypedDict):
    task_id:        str
    university:     str                # e.g. "mumbai" or "generic"
    pdf_path:       str
    image_paths:    List[str]
    raw_text:       str
    clean_text:     str
    header:         Dict[str, str]
    body_text:      str
    normalized:     str
    structured:     Dict[str, Any]
    diagram_paths:  List[str]
    flat_rows:      List[List[str]]
    excel_path:     str
    validation_ok:  bool
    retry_count:    int
    messages:       Annotated[List[Any], operator.add]
    errors:         List[str]


# --------------------------------------------------------------------------
# HEADER EXTRACTION — UNIVERSITY-AWARE
# --------------------------------------------------------------------------

def _extract_header_generic(text: str) -> Dict[str, str]:
    h = dict(paper_code="", subject="", date="", exam="",
             max_marks="", time="", qp_code="",
             semester="", scheme="", branch="", note="")
    if m := re.search(r"Subject\s*Code[:\s]+([A-Za-z0-9\-]+)", text, re.I):
        h["paper_code"] = m.group(1).strip()
    if m := re.search(r"Subject\s*Code[:\s]+[A-Za-z0-9\-]+\s*/\s*(.*?)(?:\n|\d{2}/\d{2}/\d{4})", text, re.I):
        h["subject"] = m.group(1).strip().rstrip("/").strip()
    if m := re.search(r"\d{2}/\d{2}/\d{4}", text):
        h["date"] = m.group()
    if m := re.search(r"Max\.?\s*Marks[:\s]+(\d+)", text, re.I):
        h["max_marks"] = m.group(1)
    if m := re.search(r"Time[:\s]+([0-9\.]+\s*hours?)", text, re.I):
        h["time"] = m.group(1)
    if m := re.search(r"QP\s*CODE[:\s]+(\d+)", text, re.I):
        h["qp_code"] = m.group(1)
    if m := re.search(r"Note[:\s]+(.+?)(?:\n|$)", text, re.I):
        h["note"] = m.group(1).strip()
    return h


def _extract_header_mumbai(text: str) -> Dict[str, str]:
    """
    Tuned for Mumbai University question paper format:
    Paper / Subject Code: 37473 / Software Engineering and Project Management
    10/12/2024 CSE-AIML SEM-VI C SCHEME SEPM QP CODE: 10064710
    Time: 3 hour   Max. Marks: 80
    Note: Question 1 is compulsory. Attempt any 3 out of remaining 5 questions.
    """
    h = dict(paper_code="", subject="", date="", exam="",
             max_marks="", time="", qp_code="",
             semester="", scheme="", branch="", note="")

    # Paper code & subject name from "Paper / Subject Code: XXXXX / Subject Name"
    psc = re.search(
        r"(?:Paper\s*/\s*)?Subject\s*Code[:\s]+([A-Za-z0-9\-]+)\s*/\s*(.*?)(?:\n|$)",
        text, re.I
    )
    if psc:
        h["paper_code"] = psc.group(1).strip()
        h["subject"]    = psc.group(2).strip().rstrip("/").strip()

    # Date — DD/MM/YYYY
    if m := re.search(r"\d{2}/\d{2}/\d{4}", text):
        h["date"] = m.group()

    # Branch e.g. CSE-AIML, CE, IT, EXTC
    if m := re.search(r"\b(CSE[-\s]?[A-Z]+|CE|IT|EXTC|MECH|CIVIL|AIDS|DS)\b", text, re.I):
        h["branch"] = m.group(1).upper()

    # Semester e.g. SEM-VI, SEM V, SEMVI
    if m := re.search(r"SEM[-\s]?([IVX]+|\d+)", text, re.I):
        h["semester"] = "SEM-" + m.group(1).strip().upper()

    # Scheme e.g. C SCHEME, C-SCHEME, D SCHEME
    if m := re.search(r"([A-D])[-\s]?SCHEME", text, re.I):
        h["scheme"] = m.group(1).upper() + " SCHEME"

    # QP Code
    if m := re.search(r"QP\s*CODE[:\s]+(\d+)", text, re.I):
        h["qp_code"] = m.group(1)

    # Time
    if m := re.search(r"Time[:\s]+([0-9\.]+\s*(?:hours?|hrs?))", text, re.I):
        h["time"] = m.group(1).strip()

    # Max Marks
    if m := re.search(r"Max\.?\s*Marks[:\s]+(\d+)", text, re.I):
        h["max_marks"] = m.group(1)

    # Note / Instructions
    if m := re.search(r"Note[:\s]+(.+?)(?:\n(?:Q|$)|$)", text, re.I | re.DOTALL):
        raw_note = m.group(1).strip().replace("\n", " ")
        h["note"] = raw_note[:300]

    return h


def _extract_header(text: str, university: str = "generic") -> Dict[str, str]:
    if university == "mumbai":
        return _extract_header_mumbai(text)
    return _extract_header_generic(text)


# --------------------------------------------------------------------------
# DEEP HIERARCHICAL STRUCTURE PARSING VIA GROQ LLM (JSON SCHEMA)
# --------------------------------------------------------------------------

STRUCTURE_PROMPT = """You are an expert academic document parser specializing in university exam papers, specifically in engineering, mathematics, computer science, and technology.

Given the following OCR-extracted text from a university exam paper body (the part AFTER the header), extract ALL questions into a perfectly structured JSON object.

CRITICAL PARSING & TUNING RULES:
1. QUESTION HIERARCHICAL STRUCTURE:
   - Questions follow a strict nested hierarchy:
     - Level 1 (Main Questions): Q1, Q2, Q3, Q4, Q5, Q6...
     - Level 2 (Sub-questions): A, B, C, D  OR  a, b, c, d  OR  1, 2, 3, 4 (whichever is used)
     - Level 3 (Sub-sub-questions): i, ii, iii, iv  OR  a, b, c  OR  1, 2, 3 (nested inside Level 2)
     - Level 4 (Sub-sub-sub-questions): a, b  OR  i, ii  (nested inside Level 3)
2. TABULAR DATA IN EXAMS:
   - If a question contains a table (e.g., state transition table, truth table, data rows, cost matrices, analysis points), format the table beautifully as a clean markdown table (e.g., | State | Input | Next State |) within the "text" field of that question.
3. MATHEMATICAL EQUATIONS & TECHNICAL FORMULAS:
   - Clean and format mathematical symbols (e.g., theta, lambda, delta, pi, summation, integral, limits), exponents (e.g., x^2, e^-t), fractions (e.g., 1/2), and matrices. Use standard math notation or inline LaTeX ($...$) to ensure it reads cleanly and professionally. Fix split characters (like 'x 2' to 'x^2' or 'd/d t' to 'd/dt').
4. CIRCUITS, NETWORK DIAGRAMS, AND VISUALS:
   - Keep complete references to figures, schematics, circuit models, truth tables, or UML specifications (e.g. "For the circuit shown in Fig. 1...", "Find the transfer function for the network..."). Do not strip the visual descriptors.
5. NODE CONTENT SCHEMA:
   - Each node must contain: "text" (the question text, formatted with clean tables/equations if any), "marks" (integer or null if not found), "subs" (object of child nodes, empty {} if none).
   - Marks are written as (5), [5], 5 marks, [10 marks] or sometimes at the end of a line. Extract only the number.
   - If a question part is split across lines due to OCR noise, merge them into a single coherent paragraph.
   - Return ONLY valid JSON. No markdown code fences, no explanation, just raw JSON.
6. CRITICAL SPECIAL CASE — "Write short note on any N" / Comma-listed topics:
   - ANY question that says "write short note on any N", "explain any N of the following", "attempt any N from", "write notes on any two", or lists topics after a colon like "Question: TopicA, TopicB, TopicC" MUST be split.
   - The parent node text = the instruction part only (e.g. "Write short note on any 2 from the following:")
   - EACH topic after the colon becomes its OWN child sub-node: "a", "b", "c", etc.
   - Each child: { "text": "<topic name>", "marks": <parent_marks / N if calculable, else null>, "subs": {} }
   - NEVER flatten a comma-separated topic list into one text field. This is the most common parsing error — avoid it.
7. INLINE SUB-PART LISTS (a) ... (b) ... (c) ... on the same line:
   - If a question includes inline labeled parts like "a) TopicA  b) TopicB" or "(i) ... (ii) ..."
   - Split each into its own sub-node even if they appear on the same line in the OCR text.

REQUIRED OUTPUT FORMAT (example):
{
  "Q1": {
    "text": "Answer all of the following.",
    "marks": null,
    "subs": {
      "A": {
        "text": "Find the truth table for the logic gate circuit shown below.",
        "marks": 5,
        "subs": {}
      },
      "B": {
        "text": "Solve the differential equation: $d^2y/dx^2 + 5dy/dx + 6y = 0$ given $y(0)=1$.",
        "marks": 10,
        "subs": {}
      }
    }
  },
  "Q5": {
    "text": "Write short note on any 2 from the following:",
    "marks": 10,
    "subs": {
      "a": { "text": "Reverse Engineering Process", "marks": 5, "subs": {} },
      "b": { "text": "Unit Testing and Integration Testing", "marks": 5, "subs": {} },
      "c": { "text": "Software Design Patterns", "marks": 5, "subs": {} }
    }
  }
}

TEXT TO PARSE:
"""


def _build_tree_via_llm(text: str) -> Dict:
    """Use Groq key pool to produce a fully nested JSON question tree."""
    prompt = STRUCTURE_PROMPT + text[:7000]

    for attempt in range(6):  # try each key once
        llm, api_key = groq_pool.get_llm()
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw  = resp.content.strip()
            # Strip markdown code fences if present
            raw  = re.sub(r'^```(?:json)?\s*', '', raw)
            raw  = re.sub(r'\s*```$', '', raw)
            tree = json.loads(raw)
            return tree
        except json.JSONDecodeError as e:
            print(f"  [Groq Structure] JSON parse failed (attempt {attempt+1}): {e}")
            return {}  # bad JSON -- don't retry with other keys, same output
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower() or 'token' in err.lower():
                print(f"  [Groq Structure] Key {api_key[:12]}... rate/daily-limited. Trying next key.")
                groq_pool.handle_error(api_key, err)
                continue
            print(f"  [Groq Structure] Error: {e}")
            return {}

    print("  [Groq Structure] All 6 keys exhausted/rate-limited.")
    return {}


def _flatten_tree(tree: Dict, path: List = None, rows: List = None) -> List:
    """Recursively flatten nested tree into tabular rows with 4 hierarchy levels."""
    if path is None:
        path = []
    if rows is None:
        rows = []
    for key, node in tree.items():
        new_path = path + [key]
        # Pad / truncate to 4 hierarchy columns: Q, Sub, Sub-Sub, Sub-Sub-Sub
        h = (new_path + [""] * 4)[:4]
        marks_val = str(node.get("marks", "") or "")
        rows.append(h + [node.get("text", ""), marks_val, ""])
        if node.get("subs"):
            _flatten_tree(node["subs"], new_path, rows)
    return rows


# --------------------------------------------------------------------------
# BASIC IMAGE PRE-PROCESSING HELPERS
# --------------------------------------------------------------------------

def _preprocess_for_ocr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold for better OCR on varied backgrounds
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    return thresh


def _clean_text(text: str) -> str:
    cleaned = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "-----" in line or "=====" in line:
            continue
        noise = len(re.findall(r"[^a-zA-Z0-9\s\.\)\(:/-]", line))
        if noise > len(line) * 0.45:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# --------------------------------------------------------------------------
# GRAPH NODES
# --------------------------------------------------------------------------

def pdf_ingestion_node(state: QPState) -> QPState:
    task_id = state.get("task_id", "default")
    update_task_progress(task_id, "pdf_ingestion", 10, "Rendering PDF pages with PyMuPDF...")
    try:
        doc = fitz.open(state["pdf_path"])
        image_paths = []
        for i, page in enumerate(doc):
            # 200 DPI for sharper OCR results
            pix = page.get_pixmap(dpi=200)
            path = os.path.join(OUTPUT_DIR, f"{task_id}_page_{i+1}.png")
            pix.save(path)
            image_paths.append(path)
        msg = f"PDF Ingestion complete: {len(image_paths)} page(s) rendered."
        update_task_progress(task_id, "pdf_ingestion", 20, msg)
        return {**state, "image_paths": image_paths,
                "messages": state["messages"] + [AIMessage(content=msg)]}
    except Exception as e:
        err = f"pdf_ingestion failed: {e}"
        update_task_progress(task_id, "pdf_ingestion", 20, f"Error: {err}")
        return {**state, "errors": state["errors"] + [err]}


def ocr_node(state: QPState) -> QPState:
    task_id = state.get("task_id", "default")
    update_task_progress(task_id, "ocr", 30, "Running Tesseract OCR on page images...")
    if not state.get("image_paths"):
        return {**state, "errors": state["errors"] + ["ocr: no image paths"]}

    full_text = ""
    for i, path in enumerate(state["image_paths"]):
        update_task_progress(task_id, "ocr", 30 + int(i * 8 / len(state["image_paths"])),
                             f"Analyzing page {i+1}...")
        thresh = _preprocess_for_ocr(path)
        text = pytesseract.image_to_string(thresh, config=OCR_CONFIG)
        full_text += text + "\n"
        print(f"  Page {i+1}: {len(text)} chars")

    msg = f"OCR complete. Total chars: {len(full_text)}."
    update_task_progress(task_id, "ocr", 40, msg)
    return {**state, "raw_text": full_text,
            "messages": state["messages"] + [AIMessage(content=msg)]}


def cleaning_node(state: QPState) -> QPState:
    task_id = state.get("task_id", "default")
    update_task_progress(task_id, "cleaning", 45, "Cleaning OCR noise and artifacts...")
    cleaned = _clean_text(state.get("raw_text", ""))

    if len(cleaned) > 100:
        update_task_progress(task_id, "cleaning", 50, "Groq LLM secondary cleaning pass...")
        prompt = (
            "You are an OCR text cleaning assistant for university exam papers in technical disciplines (Engineering, Math, Physics).\n"
            "1. Fix broken words, merge lines split due to layout columns, and remove page footers, watermarks, or garbage characters.\n"
            "2. Standardize mathematical equations: clean and repair scientific/technical symbols (theta, delta, summation, integrations, pi, exponents, subscripts, vectors) and matrices.\n"
            "3. Preserve all structural elements including tables, truth tables, list headers, and notes.\n"
            "4. CRITICAL: Preserve ALL question numbers (Q1, Q2...), sub-question labels (A, B, C, a, b, 1, 2, i, ii...), marks notation, and the header block exactly as found.\n"
            "Return ONLY the cleaned text. No preamble or explanations.\n\n"
            f"TEXT:\n{cleaned[:7000]}"
        )
        for attempt in range(6):
            llm, api_key = groq_pool.get_llm()
            try:
                resp    = llm.invoke([HumanMessage(content=prompt)])
                cleaned = resp.content.strip()
                update_task_progress(task_id, "cleaning", 55, "Groq LLM cleaning complete.")
                break
            except Exception as e:
                err = str(e)
                if '429' in err or 'rate' in err.lower() or 'token' in err.lower():
                    print(f"  [Cleaning] Key {api_key[:12]}... rate/daily-limited. Trying next.")
                    groq_pool.handle_error(api_key, err)
                    continue
                update_task_progress(task_id, "cleaning", 55, f"LLM cleaning skipped: {e}")
                break
        else:
            update_task_progress(task_id, "cleaning", 55, "LLM cleaning skipped -- all keys rate-limited.")

    update_task_progress(task_id, "cleaning", 60, f"Cleaning done. {len(cleaned)} chars retained.")
    return {**state, "clean_text": cleaned,
            "messages": state["messages"] + [AIMessage(content=f"Cleaning done. {len(cleaned)} chars.")]}


def header_extraction_node(state: QPState) -> QPState:
    task_id    = state.get("task_id", "default")
    university = state.get("university", "generic")
    update_task_progress(task_id, "header_extraction", 65,
                         f"Extracting header ({university.title()} University mode)...")

    text   = state.get("clean_text", "")
    header = _extract_header(text, university)

    # Separate body text — everything after the first Q1/Q.1
    split = re.split(r"(?:^|\n)\s*(Q\.?\s*1\b)", text, maxsplit=1, flags=re.I | re.M)
    if len(split) > 1:
        body = (split[1] + split[2]).strip() if len(split) > 2 else split[1].strip()
    else:
        body = text

    update_task_progress(task_id, "header_extraction", 70,
                         f"Header extracted. Subject: {header.get('subject') or 'Generic Subject'}")
    return {**state, "header": header, "body_text": body,
            "messages": state["messages"] + [AIMessage(content=f"Header: {header}")]}


def normalization_node(state: QPState) -> QPState:
    task_id = state.get("task_id", "default")
    update_task_progress(task_id, "normalization", 72,
                         "Normalizing structural boundaries for hierarchical parsing...")
    # Light normalization — just ensure Q markers have a preceding newline
    text = state.get("body_text", "")
    text = re.sub(r"(?<!\n)(Q\.?\s*\d+)", r"\n\1", text, flags=re.I)
    update_task_progress(task_id, "normalization", 78, "Normalization complete.")
    return {**state, "normalized": text,
            "messages": state["messages"] + [AIMessage(content="Normalization done.")]}


def structure_node(state: QPState) -> QPState:
    task_id = state.get("task_id", "default")
    update_task_progress(task_id, "structure", 80,
                         "Invoking Groq Llama-3.3-70b for deep hierarchical JSON tree construction...")

    normalized = state.get("normalized", "")

    # Always use Groq for the deep parse
    tree = _build_tree_via_llm(normalized)
    q_count = len(tree)
    update_task_progress(task_id, "structure", 88,
                         f"JSON tree built: {q_count} main question(s) extracted.")
    print(f"  {q_count} top-level question(s) found via Groq JSON parser.")

    return {**state, "structured": tree,
            "messages": state["messages"] + [AIMessage(content=f"Structure: {q_count} main questions.")]}


def _find_diagram_regions(img: np.ndarray) -> List[tuple]:
    """Use OpenCV to detect table grids and circuit-like regions precisely."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── 1. Detect table grid: morphological line detection ──
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // 8), 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h // 8)))
    h_lines  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    v_lines  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    grid     = cv2.add(h_lines, v_lines)

    # ── 2. Detect circuit/diagram: Canny edge density ──
    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.add(grid, edges)

    # Dilate to merge nearby blobs
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated  = cv2.dilate(combined, kernel_d, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    min_area = h * w * 0.018  # at least 1.8% of page area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Skip full-page blobs (just noise)
        if cw > w * 0.92 and ch > h * 0.92:
            continue
        # Pad by 12 pixels
        x1 = max(0, x - 12)
        y1 = max(0, y - 12)
        x2 = min(w, x + cw + 12)
        y2 = min(h, y + ch + 12)
        regions.append((x1, y1, x2, y2))
    return regions


def diagram_node(state: QPState) -> QPState:
    task_id     = state.get("task_id", "default")
    image_paths = state.get("image_paths", [])
    structured  = state.get("structured", {})
    update_task_progress(task_id, "diagram", 90, "Scanning for diagram/figure keywords...")

    keywords = {
        "diagram", "figure", "draw", "sketch", "plot", "graph", "circuit",
        "flowchart", "network", "gate", "schematic", "truth table", "waveform",
        "table", "block diagram", "circuit diagram", "karnaugh", "k-map",
        "timing diagram", "state machine", "ladder"
    }
    paths: List[str] = []

    def _has_kw(tree: Dict) -> bool:
        for node in tree.values():
            txt = node.get("text", "").lower()
            if any(kw in txt for kw in keywords):
                return True
            if node.get("subs") and _has_kw(node["subs"]):
                return True
        return False

    if image_paths and _has_kw(structured):
        for page_idx, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            regions = _find_diagram_regions(img)
            if regions:
                for reg_idx, (x1, y1, x2, y2) in enumerate(regions):
                    crop     = img[y1:y2, x1:x2]
                    out_path = os.path.join(OUTPUT_DIR, f"{task_id}_diagram_p{page_idx+1}_r{reg_idx+1}.png")
                    cv2.imwrite(out_path, crop)
                    paths.append(out_path)
        msg = f"Precise diagram/table region(s) detected and cropped: {len(paths)} image(s)."
    else:
        msg = "No diagram keywords detected — skipping visual extraction."

    update_task_progress(task_id, "diagram", 92, msg)
    return {**state, "diagram_paths": paths,
            "messages": state["messages"] + [AIMessage(content=msg)]}


def flatten_node(state: QPState) -> QPState:
    task_id = state.get("task_id", "default")
    update_task_progress(task_id, "flatten", 94, "Flattening nested tree into tabular rows...")
    rows = _flatten_tree(state.get("structured", {}))
    update_task_progress(task_id, "flatten", 96, f"Generated {len(rows)} total rows.")
    return {**state, "flat_rows": rows,
            "messages": state["messages"] + [AIMessage(content=f"Flattened: {len(rows)} rows.")]}


def excel_writer_node(state: QPState) -> QPState:
    task_id    = state.get("task_id", "default")
    university = state.get("university", "generic")
    update_task_progress(task_id, "excel_writer", 97, "Generating Excel workbook...")

    header   = state.get("header", {})
    rows     = state.get("flat_rows", [])
    diagrams = state.get("diagram_paths", [])

    wb = Workbook()
    ws = wb.active
    ws.title = "Question Paper Extraction"

    # ── Title row ──────────────────────────────────────────────
    ws.append(["EXTRACTA — AI Question Paper Extraction Report"])
    ws.append([])

    # ── Header metadata block ──────────────────────────────────
    metadata_fields = [
        ("Paper Code",    "paper_code"),
        ("Subject",       "subject"),
        ("Date",          "date"),
        ("Time",          "time"),
        ("Max Marks",     "max_marks"),
        ("QP Code",       "qp_code"),
    ]
    if university == "mumbai":
        metadata_fields += [
            ("Branch",    "branch"),
            ("Semester",  "semester"),
            ("Scheme",    "scheme"),
            ("Note",      "note"),
        ]
    else:
        metadata_fields += [("Exam", "exam")]

    for label, key in metadata_fields:
        ws.append([label, header.get(key, "")])

    ws.append([])

    # ── Column headers ─────────────────────────────────────────
    ws.append(["Q", "Sub", "Sub-Sub", "Sub-Sub-Sub", "Question Text", "Marks", "Visual"])
    start_row = ws.max_row + 1

    for row in rows:
        while len(row) < 7:
            row.append("")
        ws.append(row)

    # ── Embed diagrams ─────────────────────────────────────────
    for i, path in enumerate(diagrams):
        if os.path.exists(path):
            try:
                ws.add_image(ExcelImage(path), f"G{start_row + i}")
            except Exception as e:
                print(f"  Image embed error: {e}")

    # ── Column widths ──────────────────────────────────────────
    for col, width in [("A",8),("B",8),("C",8),("D",8),("E",90),("F",8),("G",30)]:
        ws.column_dimensions[col].width = width

    out_path = os.path.join(OUTPUT_DIR, f"{task_id}_output.xlsx")
    wb.save(out_path)
    update_task_progress(task_id, "excel_writer", 98, f"Workbook saved: {out_path}")
    return {**state, "excel_path": out_path,
            "messages": state["messages"] + [AIMessage(content=f"Excel: {out_path}")]}


def validation_node(state: QPState) -> QPState:
    task_id = state.get("task_id", "default")
    update_task_progress(task_id, "validation", 99, "Validating extraction results...")

    ok    = bool(state.get("flat_rows")) and os.path.exists(state.get("excel_path", ""))
    retry = state.get("retry_count", 0)

    if ok:
        msg = f"Validation PASSED — {len(state['flat_rows'])} rows extracted successfully."
    else:
        msg = f"Validation FAILED — will retry (attempt {retry+1})."

    summary = ""
    if LLM_AVAILABLE and ok:
        try:
            log = "\n".join(m.content for m in state["messages"] if hasattr(m, "content"))
            resp = LLM.invoke([HumanMessage(
                content="Summarise this exam extraction in 2 concise sentences covering subject, question count, and status:\n\n" + log[-2000:]
            )])
            summary = resp.content
        except Exception:
            pass

    update_task_progress(task_id, "validation", 100, msg)
    return {**state,
            "validation_ok": ok,
            "retry_count":   retry + (0 if ok else 1),
            "messages": state["messages"] + [AIMessage(content=summary or msg)]}


# --------------------------------------------------------------------------
# CONDITIONAL EDGE
# --------------------------------------------------------------------------

def retry_router(state: QPState) -> str:
    if state.get("validation_ok"):
        return "end"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "end"


# --------------------------------------------------------------------------
# GRAPH ASSEMBLY
# --------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(QPState)
    for name, fn in [
        ("pdf_ingestion",     pdf_ingestion_node),
        ("ocr",               ocr_node),
        ("cleaning",          cleaning_node),
        ("header_extraction", header_extraction_node),
        ("normalization",     normalization_node),
        ("structure",         structure_node),
        ("diagram",           diagram_node),
        ("flatten",           flatten_node),
        ("excel_writer",      excel_writer_node),
        ("validation",        validation_node),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("pdf_ingestion")
    g.add_edge("pdf_ingestion",     "ocr")
    g.add_edge("ocr",               "cleaning")
    g.add_edge("cleaning",          "header_extraction")
    g.add_edge("header_extraction", "normalization")
    g.add_edge("normalization",     "structure")
    g.add_edge("structure",         "diagram")
    g.add_edge("diagram",           "flatten")
    g.add_edge("flatten",           "excel_writer")
    g.add_edge("excel_writer",      "validation")
    g.add_conditional_edges("validation", retry_router, {"retry": "ocr", "end": END})
    return g


# --------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# --------------------------------------------------------------------------

def run_agent_pipeline(pdf_path: str, task_id: str, university: str = "generic") -> QPState:
    memory = MemorySaver()
    graph  = build_graph().compile(checkpointer=memory)

    initial: QPState = {
        "task_id":       task_id,
        "university":    university,
        "pdf_path":      pdf_path,
        "image_paths":   [],
        "raw_text":      "",
        "clean_text":    "",
        "header":        {},
        "body_text":     "",
        "normalized":    "",
        "structured":    {},
        "diagram_paths": [],
        "flat_rows":     [],
        "excel_path":    "",
        "validation_ok": False,
        "retry_count":   0,
        "messages":      [HumanMessage(content=f"Process: {pdf_path} [{university}]")],
        "errors":        [],
    }
    config = {"configurable": {"thread_id": f"qp-{task_id}"}}
    return graph.invoke(initial, config=config)


# ==========================================================================
# ── EXTRACTA V2.2 UPGRADE: HOLISTIC MODEL ANSWER SHEET AGENT ──────────────
# ==========================================================================

class GroqKeysPool:
    """Rotates through ext1-ext6 Groq keys loaded from .env."""
    def __init__(self):
        # Load ext1-ext6 from environment (populated by .env via dotenv)
        self.keys = [
            k for k in [
                os.environ.get("GROQ_EXT1"),
                os.environ.get("GROQ_EXT2"),
                os.environ.get("GROQ_EXT3"),
                os.environ.get("GROQ_EXT4"),
                os.environ.get("GROQ_EXT5"),
                os.environ.get("GROQ_EXT6"),
            ]
            if k  # skip any None / empty entries
        ]
        if not self.keys:
            print("[GroqKeysPool] WARNING: No ext keys found in .env. Add GROQ_EXT1..EXT6.")
        else:
            print(f"[GroqKeysPool] Loaded {len(self.keys)} extraction key(s) from .env.")
        self._index   = 0
        self._lock    = threading.Lock()
        self.cooldowns = {k: 0.0 for k in self.keys}

    def _parse_retry_seconds(self, err_msg: str) -> float:
        """Parse 'try again in Xm Y.Zs' from Groq error messages."""
        m = re.search(r'try again in (\d+)m([\d.]+)s', err_msg)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2)) + 5
        m = re.search(r'try again in ([\d.]+)s', err_msg)
        if m:
            return float(m.group(1)) + 5
        return 90.0  # safe default

    def handle_error(self, key: str, err_msg: str):
        """Intelligently set cooldown based on error type."""
        with self._lock:
            if key not in self.cooldowns:
                return
            # Daily token limit (TPD) — exhausted for the day
            if 'tokens per day' in err_msg or 'TPD' in err_msg:
                cooldown = 23 * 3600  # 23 hours
                print(f"[GroqKeysPool] Key {key[:12]}... DAILY LIMIT exhausted. Cooling 23h.")
            # Per-minute or per-request rate limit
            elif '429' in err_msg or 'rate' in err_msg.lower():
                cooldown = self._parse_retry_seconds(err_msg)
                print(f"[GroqKeysPool] Key {key[:12]}... rate-limited. Cooling {cooldown:.0f}s.")
            else:
                cooldown = 30.0
                print(f"[GroqKeysPool] Key {key[:12]}... error. Cooling 30s.")
            self.cooldowns[key] = time.time() + cooldown

    def get_llm(self):
        with self._lock:
            now = time.time()
            for _ in range(len(self.keys)):
                key = self.keys[self._index]
                self._index = (self._index + 1) % len(self.keys)
                if now >= self.cooldowns[key]:
                    return ChatGroq(model="llama-3.3-70b-versatile",
                                    temperature=0.1, api_key=key), key
            # All keys on cooldown — use least-expired one
            best_key = min(self.keys, key=lambda k: self.cooldowns[k])
            return ChatGroq(model="llama-3.3-70b-versatile",
                            temperature=0.1, api_key=best_key), best_key

    def mark_cooldown(self, key: str, seconds: float = 30.0):
        """Legacy compat — prefer handle_error() for new code."""
        with self._lock:
            if key in self.cooldowns:
                self.cooldowns[key] = time.time() + seconds
                print(f"[GroqKeysPool] Key {key[:12]}... throttled. Cool down: {seconds}s.")

groq_pool = GroqKeysPool()


def classify_question(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["differentiate", "compare", "distinguish", "difference"]):
        return "Differences"
    elif any(w in t for w in ["define", "what is", "explain", "state", "list", "write short note"]):
        return "Define"
    elif any(w in t for w in ["justify", "prove", "why", "compulsory"]):
        return "Justify"
    elif any(w in t for w in ["analyze", "solve", "compute", "calculate", "design", "diagram", "draw"]):
        return "Analysis"
    return "General Explanation"


ANSWERS_PROMPTS = {
    "Differences": (
        "Write a super-technical, university-style comparative answer for:\n"
        "Question: '{question}'\n"
        "Total Marks: {marks}\n\n"
        "STRICT OUTPUT FORMAT (PLAIN TEXT ONLY -- NO ** OR * SYMBOLS ANYWHERE):\n"
        "1. Write a 2-sentence high-level comparison paragraph.\n"
        "2. Write a comparison table with columns: | Parameter | {concept1} | {concept2} |\n"
        "   - Include exactly {num_points} distinct technical rows.\n"
        "   - Keep each cell concise (max 8 words). Every row must be on its own line.\n"
        "3. After the table, write one formal paragraph for each concept: definition, working principle, use case.\n"
        "4. Use === SECTION HEADING === for each section heading on its own line.\n"
        "5. PLAIN TEXT ONLY. No asterisks, no bold markers, no LaTeX $ symbols.\n"
        "6. Tone: strictly academic, formal, super-technical, zero emojis."
    ),
    "Define": (
        "Write a highly descriptive, super-technical university-style answer for:\n"
        "Question: '{question}'\n"
        "Total Marks: {marks}\n\n"
        "STRICT OUTPUT FORMAT (PLAIN TEXT ONLY -- NO ** OR * SYMBOLS ANYWHERE):\n"
        "1. Write a precise formal academic definition (2-3 sentences).\n"
        "2. === THEORETICAL BACKGROUND ===\n"
        "   Provide the full theoretical framework as {num_points} numbered points.\n"
        "3. === KEY CHARACTERISTICS ===\n"
        "   List {num_points} core attributes as numbered points.\n"
        "4. === APPLICATIONS ===\n"
        "   List 2-3 real-world engineering applications as numbered points.\n"
        "5. Write formulas in plain notation only: F = ma, v = u + at.\n"
        "6. PLAIN TEXT ONLY. No asterisks. UPPERCASE for emphasis only.\n"
        "7. Tone: strictly academic, formal, super-technical, zero emojis."
    ),
    "Justify": (
        "Write a highly technical, university-style justification answer for:\n"
        "Question: '{question}'\n"
        "Total Marks: {marks}\n\n"
        "STRICT OUTPUT FORMAT (PLAIN TEXT ONLY -- NO ** OR * SYMBOLS ANYWHERE):\n"
        "1. === THESIS STATEMENT ===\n"
        "   State the thesis in 1 clear sentence.\n"
        "2. === JUSTIFICATION ===\n"
        "   Provide a {num_points}-step logical argumentation.\n"
        "   Number each step. Each must be formally written and technically precise.\n"
        "3. === SUPPORTING EVIDENCE ===\n"
        "   Include formulas in plain notation, real-world evidence, or derived results.\n"
        "4. === CONCLUSION ===\n"
        "   Summarize in 2 sentences.\n"
        "5. PLAIN TEXT ONLY. No asterisks.\n"
        "6. Tone: strictly academic, formal, super-technical, zero emojis."
    ),
    "Analysis": (
        "Write a comprehensive, highly technical university-style engineering/mathematical analysis for:\n"
        "Question: '{question}'\n"
        "Total Marks: {marks}\n\n"
        "STRICT OUTPUT FORMAT (PLAIN TEXT ONLY -- NO ** OR * SYMBOLS ANYWHERE):\n"
        "1. === PROBLEM STATEMENT ===\n"
        "   Restate the problem formally in 2 sentences.\n"
        "2. === APPROACH / METHODOLOGY ===\n"
        "   Describe the method, tools, or algorithm to be used.\n"
        "3. === STEP-BY-STEP SOLUTION ===\n"
        "   Solve or design in exactly {num_points} clearly numbered steps.\n"
        "   Show all formulas in plain notation: Z = V/I, P = VI cos(phi).\n"
        "4. === RESULT / CONCLUSION ===\n"
        "   State the final result and its engineering significance.\n"
        "5. PLAIN TEXT ONLY. No asterisks, no dollar signs.\n"
        "6. Tone: strictly academic, formal, super-technical, zero emojis."
    ),
    "General Explanation": (
        "Write a formal, super-technical, highly descriptive university-style explanation for:\n"
        "Question: '{question}'\n"
        "Total Marks: {marks}\n\n"
        "STRICT OUTPUT FORMAT (PLAIN TEXT ONLY -- NO ** OR * SYMBOLS ANYWHERE):\n"
        "1. === OVERVIEW AND THEORY ===\n"
        "   Provide a 2-3 sentence formal overview.\n"
        "2. === TECHNICAL DEEP-DIVE ===\n"
        "   Provide {num_points} numbered technical points. Each must be a complete sentence.\n"
        "3. === ACADEMIC SYNTHESIS ===\n"
        "   Synthesize in 3-4 sentences covering significance, limitations, and real-world relevance.\n"
        "4. PLAIN TEXT ONLY. No asterisks. UPPERCASE for emphasis.\n"
        "5. Write formulas in plain notation: E = mc^2, F = G*m1*m2/r^2.\n"
        "6. Tone: strictly academic, formal, super-technical, zero emojis."
    )
}



def _marks_to_points(marks) -> int:
    """Return number of answer points appropriate for given marks."""
    try:
        m = int(marks or 0)
    except (ValueError, TypeError):
        m = 0
    if m <= 0:   return 6
    if m <= 3:   return 4
    if m <= 5:   return 7
    if m <= 7:   return 9
    if m <= 10:  return 12
    return 14


def solve_single_question(item: Dict[str, Any]) -> Dict[str, Any]:
    key      = item["key"]
    text     = item["text"]
    marks    = item.get("marks", 0)
    category = classify_question(text)
    num_pts  = _marks_to_points(marks)

    # Detect the two concepts for Differences prompts
    concepts = re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    concept1 = concepts[0] if len(concepts) > 0 else "Concept A"
    concept2 = concepts[1] if len(concepts) > 1 else "Concept B"

    prompt = ANSWERS_PROMPTS[category].format(
        question   = text,
        marks      = marks or "Not specified",
        num_points = num_pts,
        concept1   = concept1,
        concept2   = concept2,
    )

    for attempt in range(5):
        llm, api_key = groq_pool.get_llm()
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            return {
                "key":      key,
                "text":     text,
                "marks":    marks,
                "category": category,
                "answer":   resp.content.strip()
            }
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate" in err_msg.lower() or "token" in err_msg.lower():
                groq_pool.handle_error(api_key, err_msg)
                continue
            else:
                time.sleep(2)

    return {
        "key":      key,
        "text":     text,
        "marks":    marks,
        "category": category,
        "answer":   "Model Answer could not be generated due to service limits."
    }


def generate_answers_for_tree(structured_tree: Dict, task_id: str) -> List[Dict[str, Any]]:
    flat_questions = []

    def traverse(node_map: Dict, path: str = ""):
        for k, v in node_map.items():
            current_key = f"{path}.{k}" if path else k
            if v.get("text"):
                flat_questions.append({
                    "key":   current_key,
                    "text":  v["text"],
                    "marks": v.get("marks", 0)
                })
            if v.get("subs"):
                traverse(v["subs"], current_key)

    traverse(structured_tree)

    if not flat_questions:
        return []

    print(f"[{task_id}] Solving {len(flat_questions)} questions in parallel with 6 Groq Keys...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(solve_single_question, item): item for item in flat_questions}
        for index, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            results.append(res)
            percentage = 10 + int((index + 1) * 80 / len(flat_questions))
            update_task_progress(task_id, "answering", percentage,
                                 f"Solved {index+1} of {len(flat_questions)}: Question {res['key']}")

    results.sort(key=lambda x: x["key"])
    return results


# ── FPDF2 PDF Compiler ──

def sanitize_for_pdf(text: str) -> str:
    replacements = {
        "\u2014": " -- ", # em dash
        "\u2013": " - ",  # en dash
        "\u201c": '"',    # smart opening double quote
        "\u201d": '"',    # smart closing double quote
        "\u2018": "'",    # smart opening single quote
        "\u2019": "'",    # smart closing single quote
        "\u2208": " in ",
        "\u222b": "Integral ",
        "\u03b8": "theta",
        "\u03c0": "pi",
        "\u03bb": "lambda",
        "\u03b4": "delta",
        "\u2211": "Sum",
        "\u2192": " -> ",
        "\u2260": " != ",
        "\u2264": " <= ",
        "\u2265": " >= ",
        "\u00b1": " +/- ",
        "\u00d7": " x ",
        "\u00f7": " / ",
        "\u221e": " infinity ",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


class ModelAnswersPDF(FPDF):
    def __init__(self, subject_title: str = "EXAMINATION"):
        super().__init__()
        self.subject_title = sanitize_for_pdf(subject_title)

    def header(self):
        self.set_fill_color(15, 23, 42)
        self.rect(0, 0, 210, 14, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("times", "B", 9)
        self.set_xy(0, 2)
        self.cell(210, 10, "UNIVERSITY MODEL ANSWER SHEET -- EXTRACTA AI", align="C")
        self.ln(16)

    def footer(self):
        self.set_y(-14)
        self.set_font("times", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _strip_inline_md(text: str) -> str:
    """Remove inline markdown markers like **bold**, *italic*, and $LaTeX$ cleanly."""
    # Strip $...$ LaTeX (inline)
    text = re.sub(r'\$\$([^$]*)\$\$', lambda m: f'  {m.group(1).strip()}  ', text)
    text = re.sub(r'\$([^$\n]*)\$',   lambda m: f' {m.group(1).strip()} ',  text)
    # Strip bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*',     r'\1', text)
    # Strip backtick code spans
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text


def _is_section_heading(line: str) -> bool:
    """Detect === SECTION === headings produced by the LLM."""
    return line.startswith("===") and line.endswith("===")


def write_markdown_to_pdf(pdf: FPDF, text: str):
    text = sanitize_for_pdf(text)
    lines = text.split("\n")
    in_table = False
    table_data = []
    line_h = 5

    pdf.set_font("times", "", 10)
    pdf.set_text_color(30, 41, 59)

    def _flush_table():
        nonlocal in_table, table_data
        if table_data:
            _render_table(pdf, table_data)
        table_data = []
        in_table = False

    for line in lines:
        line = line.strip()

        if not line:
            if in_table:
                _flush_table()
            pdf.ln(2)
            continue

        # === SECTION HEADING ===
        if _is_section_heading(line):
            if in_table:
                _flush_table()
            heading = line.strip("= ").strip()
            pdf.set_font("times", "B", 11)
            pdf.set_fill_color(230, 235, 248)
            pdf.set_text_color(40, 50, 120)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 7, sanitize_for_pdf(heading), fill=True)
            pdf.ln(8)
            pdf.set_font("times", "", 10)
            pdf.set_text_color(30, 41, 59)
            continue

        # Markdown ## / # headings
        if line.startswith("### "):
            if in_table: _flush_table()
            heading = _strip_inline_md(line[4:])
            pdf.set_font("times", "B", 11)
            pdf.set_text_color(79, 70, 229)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 7, sanitize_for_pdf(heading))
            pdf.ln(8)
            pdf.set_font("times", "", 10)
            pdf.set_text_color(30, 41, 59)
            continue
        elif line.startswith("## ") or line.startswith("# "):
            if in_table: _flush_table()
            heading = _strip_inline_md(line[3:] if line.startswith("## ") else line[2:])
            pdf.set_font("times", "B", 12)
            pdf.set_text_color(60, 80, 200)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 8, sanitize_for_pdf(heading))
            pdf.ln(9)
            pdf.set_font("times", "", 10)
            pdf.set_text_color(30, 41, 59)
            continue

        # Markdown Tables
        if line.startswith("|"):
            in_table = True
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if parts and all(set(p) <= set("-: ") for p in parts):
                continue  # skip separator row
            table_data.append([_strip_inline_md(p) for p in parts])
            continue

        if in_table and not line.startswith("|"):
            _flush_table()

        # Numbered list: 1. 2. 3.
        if re.match(r'^\d+\.\s', line):
            clean = sanitize_for_pdf(_strip_inline_md(line))
            pdf.set_font("times", "", 10)
            pdf.set_text_color(30, 41, 59)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(0, line_h, clean)
            pdf.ln(1)
            continue

        # Bullet points: - or *
        if re.match(r'^[\-\*\+]\s', line):
            clean = sanitize_for_pdf(_strip_inline_md(line[2:]))
            pdf.set_font("times", "", 10)
            pdf.set_text_color(30, 41, 59)
            pdf.set_x(pdf.l_margin + 2)
            pdf.write(line_h, "-  ")
            pdf.multi_cell(0, line_h, clean)
            pdf.ln(1)
            continue

        # Bold-only lines (lines that were entirely **bold** headings from LLM)
        is_bold = re.match(r'^\*\*(.+)\*\*$', line)
        if is_bold:
            clean = sanitize_for_pdf(is_bold.group(1))
            pdf.set_font("times", "B", 10)
            pdf.set_text_color(30, 41, 59)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, line_h, clean)
            pdf.set_font("times", "", 10)
            pdf.ln(1)
            continue

        # Regular paragraph text
        clean = sanitize_for_pdf(_strip_inline_md(line))
        pdf.set_font("times", "", 10)
        pdf.set_text_color(30, 41, 59)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, line_h, clean)
        pdf.ln(1)

    if in_table and table_data:
        _flush_table()


def _render_table(pdf: FPDF, rows: List[List[str]]):
    """Render a markdown table with auto-height cells so text never overlaps."""
    if not rows:
        return

    col_count = max(len(r) for r in rows)
    page_w    = pdf.w - pdf.l_margin - pdf.r_margin
    col_w     = page_w / col_count
    line_h    = 5
    x_start   = pdf.l_margin

    for row_idx, row in enumerate(rows):
        padded = (row + [""] * col_count)[:col_count]
        sanitized = [sanitize_for_pdf(_strip_inline_md(str(c))) for c in padded]

        is_header = (row_idx == 0)
        if is_header:
            pdf.set_font("times", "B", 9)
            pdf.set_fill_color(215, 225, 248)
            pdf.set_text_color(15, 23, 42)
        else:
            pdf.set_font("times", "", 9)
            fill_r = (245, 247, 252) if row_idx % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*fill_r)
            pdf.set_text_color(30, 41, 59)

        y_start = pdf.get_y()

        # Page-break guard before each row
        if y_start > pdf.h - pdf.b_margin - 25:
            pdf.add_page()
            y_start = pdf.get_y()

        # First pass: measure max lines needed in this row
        max_lines = 1
        for cell_text in sanitized:
            sw = pdf.get_string_width(cell_text)
            avail = col_w - 4
            n_lines = max(1, int(sw / max(avail, 1)) + 1)
            max_lines = max(max_lines, n_lines)
        row_h = max_lines * line_h + 3

        # Second pass: draw each cell at the same y_start, track max y
        max_y = y_start
        for ci, cell_text in enumerate(sanitized):
            x = x_start + ci * col_w
            pdf.set_xy(x, y_start)
            pdf.multi_cell(col_w, line_h, cell_text, border=1, align="L", fill=True)
            max_y = max(max_y, pdf.get_y())

        # Advance to the lowest cell bottom, add tiny gap
        pdf.set_xy(x_start, max_y)

    pdf.ln(4)


def compile_answers_pdf(results: List[Dict[str, Any]], header: Dict[str, str], task_id: str) -> str:
    subject = header.get("subject", "Academic Examination")
    pdf = ModelAnswersPDF(subject_title=subject)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Title ──────────────────────────────────────────────────────
    pdf.set_font("times", "B", 15)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 10, sanitize_for_pdf(f"Model Answer Sheet: {subject}"), align="C")
    pdf.ln(3)

    # ── Metadata box (simple two-column rows using set_x + cell + ln) ──
    def meta_row(label: str, value: str):
        pdf.set_font("times", "B", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.set_x(18)
        pdf.cell(55, 6, sanitize_for_pdf(label))
        pdf.set_font("times", "", 9)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(120, 6, sanitize_for_pdf(value))
        pdf.ln(6)

    y0 = pdf.get_y()
    pdf.set_fill_color(248, 250, 252)
    pdf.rect(15, y0, 180, 42, "F")

    meta_row("Subject:",          header.get("subject", "--"))
    meta_row("Paper Code:",       header.get("paper_code", "--"))
    meta_row("QP Code:",          header.get("qp_code", "--"))
    meta_row("Semester / Scheme:",
             f"{header.get('semester', '--')} / {header.get('scheme', '--')}")
    meta_row("Branch:",           header.get("branch", "--"))
    meta_row("Time & Marks:",
             f"{header.get('time', '--')}  |  Max Marks: {header.get('max_marks', '--')}")
    meta_row("Date:",
             header.get("date", "") or time.strftime("%d/%m/%Y"))

    pdf.ln(10)

    # ── Question Answer Blocks ──────────────────────────────────────
    for item in results:
        key  = item["key"]
        text = item["text"]
        cat  = item["category"]
        ans  = item["answer"]

        # Question heading bar
        pdf.set_font("times", "B", 11)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_text_color(15, 23, 42)
        heading = sanitize_for_pdf(f"Question {key}  [{cat.upper()}]")
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 8, heading, border="TB", align="L", fill=True)
        pdf.ln(9)

        # Question text in italics
        pdf.set_font("times", "I", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(0, 5, sanitize_for_pdf(f'"{text}"'))
        pdf.ln(3)

        # Model Answer label
        pdf.set_font("times", "B", 10)
        pdf.set_text_color(79, 70, 229)
        pdf.cell(0, 6, "MODEL SOLUTION:")
        pdf.ln(7)

        # Answer content (markdown-aware)
        write_markdown_to_pdf(pdf, ans)
        pdf.ln(8)

    out_path = os.path.join(OUTPUT_DIR, f"{task_id}_answers.pdf")
    pdf.output(out_path)
    return out_path

