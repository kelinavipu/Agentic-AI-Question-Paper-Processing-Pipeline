# Extracta Pipeline v2.2

Extracta is an AI-powered pipeline designed to convert complex, bilingual, and hierarchical university question papers (PDFs) into structured Excel worksheets with generated answers.

## Groq Model Load-Balancing Hierarchy

To process 10-12 papers in bulk without hitting Groq API daily rate limits, the pipeline implements a hybrid load-balancing architecture. It routes token-heavy text processing to high-limit models and reserves the smartest model strictly for complex JSON structuring.

*   **LLM Cleaning pass:** `meta-llama/llama-4-scout-17b-16e-instruct`
    *   *Why:* This node runs for every single page in a paper. By offloading it to the 17B model (which has a massive 500,000 Tokens/Day limit), we save tens of thousands of tokens from the 70B model's daily limit.
*   **Answer Generation:** `meta-llama/llama-4-scout-17b-16e-instruct`
    *   *Why:* Generating detailed academic answers consumes up to 15,000+ tokens per paper. The 17B model is highly capable of answering academic questions and its 500K daily limit can easily handle bulk processing 10-12 papers.
*   **JSON Structuring:** `llama-3.3-70b-versatile`
    *   *Why:* Parsing complex, messy OCR text into a strict hierarchical JSON tree requires maximum reasoning intelligence. This model is reserved strictly for this step, consuming only ~5,000 tokens per paper, which easily fits within its strict 100,000 Tokens/Day limit across bulk runs.

## Pipeline Execution Steps

The extraction process is executed in a sequential LangGraph pipeline. Below are the steps and the tools used to achieve them:

1.  **Ingesting PDF (`PyMuPDF / fitz`)**
    *   Reads the uploaded PDF and renders each page into a high-resolution image. It automatically detects landscape booklets and slices them precisely down the middle to isolate columns and prevent cross-reading.
2.  **Running OCR (`Tesseract OCR / pytesseract`)**
    *   Scans the isolated page images and extracts the raw ASCII text using Tesseract.
3.  **LLM Cleaning (`Groq / Llama 4 Scout 17B`)**
    *   Processes the raw text to remove noise. Crucially, it deletes garbled Hindi OCR characters while preserving question numbers (e.g., `(i)`) and deduplicating multiple-choice options.
4.  **Header Extraction (`Regex Pattern Matching`)**
    *   Extracts critical metadata like Subject, Paper Code, Max Marks, Time, and Date directly from the raw OCR text using regular expressions.
5.  **Normalization (`Python String Manipulation`)**
    *   Prepares and sanitizes structural boundaries in the text to assist the LLM in understanding the hierarchy.
6.  **Groq JSON Structuring (`Groq / Llama 3.3 70B`)**
    *   The core brain of the pipeline. It reads the cleaned text and outputs a deeply nested JSON tree categorizing questions, sub-questions, MCQs, options, and marks.
7.  **Cropping Visuals (`OpenCV / cv2` & `PyMuPDF`)**
    *   Scans the text for visual keywords (e.g., "diagram", "table"). If found, it uses contour detection to locate the visual elements on the page image and crops them as separate image files.
8.  **Flattening Data (`Recursive Python Function`)**
    *   Recursively walks the nested JSON tree and flattens it into a 2D tabular format (rows and columns) suitable for a spreadsheet. It defaults missing types to "descriptive" to prevent crashes.
9.  **Writing Excel (`openpyxl`)**
    *   Takes the flattened tabular data and generates a cleanly formatted `.xlsx` workbook.
10. **Verifying (`Validation Logic`)**
    *   Checks the final extraction to ensure questions were successfully captured. If the extraction failed (e.g., 0 questions extracted), it automatically triggers a retry using the next available API key in the pool.
