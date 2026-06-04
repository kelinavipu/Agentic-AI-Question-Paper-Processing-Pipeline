import os
import uuid
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env before importing agent so all keys are available
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except ImportError:
    pass

import agent
from agent import run_agent_pipeline, task_progress_logs, LLM, LLM_AVAILABLE
from langchain_core.messages import HumanMessage

app = FastAPI(title="Extracta — AI Question Paper Extraction Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

tasks_db: dict = {}

# --------------------------------------------------------------------------
# FILENAME SLUG HELPER
# --------------------------------------------------------------------------

def _make_filename_slug(header: dict, task_id: str) -> str:
    """Build a human-readable filename from paper metadata."""
    import re as _re
    parts = []
    subject = header.get("subject", "") or ""
    if subject:
        clean = _re.sub(r'[^a-zA-Z0-9\s]', '', subject).strip()
        words = clean.split()
        slug  = '_'.join(w.capitalize() for w in words[:5])
        parts.append(slug)
    sem = (header.get("semester") or "").replace("-", "").replace(" ", "")
    if sem:
        parts.append(sem)
    scheme = (header.get("scheme") or "").replace(" ", "")
    if scheme:
        parts.append(scheme)
    branch = (header.get("branch") or "").replace("-", "_")
    if branch:
        parts.append(branch)
    parts.append(task_id[:6])
    return '_'.join(filter(None, parts))

# --------------------------------------------------------------------------
# BACKGROUND PIPELINE RUNNER
# --------------------------------------------------------------------------

def _run_extraction_task(task_id: str, pdf_path: str, university: str, is_extract_hindi: bool = False, original_filename: str = ""):
    try:
        tasks_db[task_id] = {"status": "processing", "progress": 5, "error": None, "result": None, "hindi_detected": False}
        final = run_agent_pipeline(pdf_path, task_id, university, is_extract_hindi, original_filename)
        
        # Pull Hindi detection flag from final state to update UI
        tasks_db[task_id]["hindi_detected"] = final.get("hindi_detected", False)

        if final.get("validation_ok"):
            tasks_db[task_id].update({
                "status": "completed",
                "progress": 100,
                "result": {
                    "header":           final.get("header", {}),
                    "structured":       final.get("structured", {}),
                    "flat_rows_count":  len(final.get("flat_rows", [])),
                    "excel_path":       final.get("excel_path", ""),
                    "diagrams":         [os.path.basename(p) for p in final.get("diagram_paths", [])],
                    "university":       final.get("university", "generic"),
                    "original_filename": final.get("original_filename", ""),
                }
            })
        else:
            errors = final.get("errors", [])
            tasks_db[task_id].update({
                "status": "failed",
                "progress": 100,
                "error": errors[-1] if errors else "Pipeline validation failed."
            })
    except Exception as e:
        tasks_db[task_id].update({"status": "failed", "progress": 100, "error": str(e)})
        task_progress_logs.setdefault(task_id, []).append({
            "node": "error", "progress": 100, "message": f"Critical error: {e}"
        })


# --------------------------------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------------------------------

@app.post("/api/extract")
async def extract_pdf(
    file: UploadFile = File(...),
    university: str  = Form("generic"),
    extract_hindi: str = Form("false")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    task_id  = uuid.uuid4().hex[:8]
    pdf_path = os.path.join(UPLOAD_DIR, f"{task_id}_input.pdf")

    with open(pdf_path, "wb") as buf:
        buf.write(await file.read())

    is_extract_hindi = extract_hindi.lower() == "true"
    original_filename = file.filename
    thread = threading.Thread(target=_run_extraction_task, args=(task_id, pdf_path, university, is_extract_hindi, original_filename))
    thread.daemon = True
    thread.start()

    return {"task_id": task_id, "status": "queued", "university": university}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found.")

    task  = tasks_db[task_id]
    logs  = task_progress_logs.get(task_id, [])

    current_progress = 0
    current_node     = "pending"
    current_message  = "Queueing extraction job..."

    if logs:
        latest           = logs[-1]
        current_progress = latest["progress"]
        current_node     = latest["node"]
        current_message  = latest["message"]

    return {
        "task_id":  task_id,
        "status":   task["status"],
        "hindi_detected": task.get("hindi_detected", False),
        "node":     current_node,
        "progress": current_progress,
        "message":  current_message,
        "logs":     logs,
        "error":    task["error"],
    }


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found.")
    task = tasks_db[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task not yet complete: {task['status']}")
    return task["result"]


@app.get("/api/download/{task_id}")
async def download_excel(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found.")
    task = tasks_db[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Excel not yet generated.")
    excel_path = task["result"]["excel_path"]
    if not os.path.exists(excel_path):
        raise HTTPException(status_code=404, detail="Excel file not found on disk.")
    header = task["result"].get("header", {})
    slug   = _make_filename_slug(header, task_id)
    return FileResponse(path=excel_path,
                        filename=os.path.basename(excel_path),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/api/media/{task_id}/{filename}")
async def get_media(task_id: str, filename: str):
    safe = os.path.basename(filename)
    if not safe.startswith(task_id):
        raise HTTPException(status_code=403, detail="Access denied.")
    fp = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(fp)


class SparkRequest(BaseModel):
    question_text: str

@app.post("/api/spark")
async def generate_spark(req: SparkRequest):
    if not LLM_AVAILABLE:
        return {"spark": "Knowledge portal is offline. Check API key configuration."}
    try:
        prompt = (
            "You are an inspiring academic mentor. A student has this exam question:\n"
            f"'{req.question_text}'\n\n"
            "Generate exactly THREE sections. STRICTLY ZERO EMOJIS. Use bold markdown headers:\n\n"
            "**Real-world Connection**: [2 sentences connecting this topic to cutting-edge technology, engineering, science, or real life]\n"
            "**Mind-blowing Fact**: [2 sentences with a fascinating historical, scientific, or mathematical fact related to this topic]\n"
            "**Curious Quest**: [1 compelling thought experiment or open question to ignite curiosity]\n\n"
            "Tone: intellectually captivating, professional, zero emojis."
        )
        resp = LLM.invoke([HumanMessage(content=prompt)])
        return {"spark": resp.content}
    except Exception as e:
        return {"spark": f"Portal error: {e}"}


# --------------------------------------------------------------------------
# EXTRACTA V2.2: HOLISTIC ANSWER GENERATION ENDPOINTS
# --------------------------------------------------------------------------

answers_db: dict = {}

def _run_answers_task(task_id: str):
    try:
        answers_db[task_id] = {"status": "processing", "pdf_path": None, "error": None}
        task_progress_logs[task_id] = [] # Reset progress logs for the answering task
        
        # 1. Fetch completed extraction
        if task_id not in tasks_db or tasks_db[task_id]["status"] != "completed":
            raise ValueError("Associated extraction task not found or not completed.")
            
        result = tasks_db[task_id]["result"]
        structured_tree = result["structured"]
        header = result["header"]
        
        # 2. Solve questions in parallel via 6 Groq keys pool
        agent.update_task_progress(task_id, "answering", 5, "Initializing 6 Groq Keys Pool Manager...")
        results = agent.generate_answers_for_tree(structured_tree, task_id)
        
        # 3. Compile report PDF
        agent.update_task_progress(task_id, "answering", 95, "Compiling premium model answer sheet PDF...")
        pdf_path = agent.compile_answers_pdf(results, header, task_id, result.get("pdf_path", ""))
        
        answers_db[task_id].update({
            "status": "completed",
            "pdf_path": pdf_path
        })
        agent.update_task_progress(task_id, "answering", 100, "Model Answer Sheet PDF generation PASSED.")
    except Exception as e:
        answers_db[task_id].update({
            "status": "failed",
            "error": str(e)
        })
        agent.update_task_progress(task_id, "answering", 100, f"Error: {e}")


@app.post("/api/answers/{task_id}")
async def generate_answers(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found.")
    
    # Run answering in background thread
    thread = threading.Thread(target=_run_answers_task, args=(task_id,))
    thread.daemon = True
    thread.start()
    
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/answers/status/{task_id}")
async def get_answers_status(task_id: str):
    if task_id not in answers_db:
        return {"status": "not_started", "progress": 0, "message": "Awaiting generation request..."}
        
    task = answers_db[task_id]
    logs = task_progress_logs.get(task_id, [])
    
    current_progress = 0
    current_message  = "Queueing answering task..."
    
    if logs:
        latest = logs[-1]
        current_progress = latest["progress"]
        current_message  = latest["message"]
        
    return {
        "status": task["status"],
        "progress": current_progress,
        "message": current_message,
        "error": task["error"]
    }


@app.get("/api/download-answers/{task_id}")
async def download_answers_pdf(task_id: str):
    if task_id not in answers_db:
        raise HTTPException(status_code=404, detail="Answer sheet not found.")
    task = answers_db[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Answer sheet PDF not yet compiled.")
    pdf_path = task["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")
    header = tasks_db.get(task_id, {}).get("result", {}).get("header", {})
    slug   = _make_filename_slug(header, task_id)
    return FileResponse(path=pdf_path,
                        filename=os.path.basename(pdf_path),
                        media_type="application/pdf")


# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("[Extracta Server v2.2] Starting on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
