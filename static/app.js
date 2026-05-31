// ── Global state ─────────────────────────────────────────────
let activeTaskId    = null;
let selectedUniv    = "mumbai";   // default
let pollInterval    = null;
let answersPollInt  = null;
let activeDiagrams  = [];         // list of diagram filenames from last processed paper

// Configure marked.js for safe rendering
if (typeof marked !== "undefined") {
    marked.setOptions({ breaks: true, gfm: true });
}

// ── Boot ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initUniversitySelector();
    initUploadZone();
    initDownloadButton();
    initAnswersButtons();
});

// ============================================================
// UNIVERSITY SELECTOR
// ============================================================

function initUniversitySelector() {
    const buttons = document.querySelectorAll(".seg-btn");
    const hint    = document.getElementById("university-hint");

    const hints = {
        mumbai:  "Tuned for Mumbai University format: QP Code, Semester, Scheme, Branch, Compulsory Note.",
        abvv:    "ABVV (Atal Bihari Vajpayee Vishwavidyalaya): SI codes, bilingual Hindi+English papers, Section A/B with MCQ + descriptive, OR alternatives.",
        generic: "Generic mode: extracts subject, code, date, marks, and time from any standard paper."
    };

    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            buttons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedUniv = btn.dataset.value;
            hint.textContent = hints[selectedUniv] || "";
        });
    });
}

// ============================================================
// UPLOAD HANDLING
// ============================================================

function initUploadZone() {
    const zone  = document.getElementById("upload-zone");
    const input = document.getElementById("file-input");

    zone.addEventListener("click", () => input.click());
    input.addEventListener("change", e => {
        if (e.target.files.length) handleUpload(e.target.files[0]);
    });
    zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", e => {
        e.preventDefault();
        zone.classList.remove("dragover");
        if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
    });
}

function handleUpload(file) {
    if (file.type !== "application/pdf") {
        alert("Only PDF question papers are supported.");
        return;
    }

    // Switch UI to processing mode
    document.getElementById("upload-zone").classList.add("hidden");
    document.getElementById("pipeline-wrapper").classList.remove("hidden");

    const logStream = document.getElementById("log-stream");
    logStream.innerHTML = `<div class="log-line system">[System] Connection established. Queueing "${file.name}"...</div>`;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("university", selectedUniv);

    appendLog("system", `University mode: ${selectedUniv.toUpperCase()}`);

    fetch("/api/extract", { method: "POST", body: formData })
        .then(r => { if (!r.ok) throw new Error("Upload failed."); return r.json(); })
        .then(data => {
            activeTaskId = data.task_id;
            appendLog("system", `Session ID: ${activeTaskId} — Pipeline starting...`);
            startPolling(activeTaskId);
        })
        .catch(err => {
            appendLog("error", `Upload error: ${err.message}`);
            setBadge("FAILED", "error");
        });
}

// ============================================================
// TELEMETRY LOG STREAM
// ============================================================

function appendLog(type, msg) {
    const stream = document.getElementById("log-stream");
    const div    = document.createElement("div");
    div.className = `log-line ${type}`;
    div.textContent = `[${type}] ${msg}`;
    stream.appendChild(div);
    stream.scrollTop = stream.scrollHeight;
}

function setBadge(text, cls) {
    const badge = document.getElementById("global-status-badge");
    badge.textContent = text;
    badge.className   = `status-badge ${cls}`;
}

// ============================================================
// AGENT GRAPH POLLING
// ============================================================

const NODE_ORDER = [
    "pdf_ingestion","ocr","cleaning","header_extraction",
    "normalization","structure","diagram","flatten","excel_writer","validation"
];

function startPolling(taskId) {
    let lastLogCount = 0;

    pollInterval = setInterval(() => {
        fetch(`/api/status/${taskId}`)
            .then(r => r.json())
            .then(data => {
                // Sync node lights
                const activeIdx = NODE_ORDER.indexOf(data.node);
                NODE_ORDER.forEach((name, idx) => {
                    const el = document.getElementById(`node-${name}`);
                    if (!el) return;
                    if (idx < activeIdx)       el.className = "graph-node completed";
                    else if (idx === activeIdx) el.className = "graph-node active";
                    else                        el.className = "graph-node";
                });

                // Append new log lines
                const logs = data.logs || [];
                for (let i = lastLogCount; i < logs.length; i++) {
                    const l = logs[i];
                    appendLog(l.node, l.message);
                }
                lastLogCount = logs.length;

                if (data.status === "completed") {
                    clearInterval(pollInterval);
                    NODE_ORDER.forEach(name => {
                        const el = document.getElementById(`node-${name}`);
                        if (el) el.className = "graph-node completed";
                    });
                    setBadge("PASSED", "success");
                    appendLog("success", "Validation passed — loading interactive explorer...");
                    loadResult(taskId);
                } else if (data.status === "failed") {
                    clearInterval(pollInterval);
                    setBadge("FAILED", "error");
                    appendLog("error", `Pipeline failed: ${data.error || "Unknown error"}`);
                }
            })
            .catch(err => console.error("Polling error:", err));
    }, 900);
}

// ============================================================
// RESULT RENDERING
// ============================================================

function loadResult(taskId) {
    fetch(`/api/result/${taskId}`)
        .then(r => { if (!r.ok) throw new Error("Failed to load results."); return r.json(); })
        .then(data => {
            document.getElementById("empty-state-explorer").classList.add("hidden");
            document.getElementById("tome-explorer").classList.remove("hidden");

            // Store diagrams globally for visual question display
            activeDiagrams = data.diagrams || [];

            renderMetadata(data.header, data.university);
            renderTree(data.structured);
        })
        .catch(err => alert("Error loading result: " + err.message));
}

function renderMetadata(h, university) {
    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val || "—";
    };

    set("meta-subject",    h.subject);
    set("meta-paper-code", h.paper_code);
    set("meta-max-marks",  h.max_marks ? h.max_marks + " marks" : "");
    set("meta-time",       h.time);
    set("meta-date",       h.date);
    set("meta-qp-code",    h.qp_code);

    // Hide all extra rows first
    document.getElementById("mumbai-extra-row")?.classList.add("hidden");
    document.getElementById("abvv-extra-row")?.classList.add("hidden");

    // Mumbai-specific extras
    if (university === "mumbai") {
        set("meta-branch",   h.branch);
        set("meta-semester", h.semester);
        set("meta-scheme",   h.scheme);
        set("meta-note",     h.note);
        document.getElementById("mumbai-extra-row").classList.remove("hidden");
    }

    // ABVV-specific extras
    if (university === "abvv") {
        set("meta-abvv-semester",  h.semester);
        set("meta-abvv-session",   h.exam_session || h.date);
        set("meta-abvv-examtype",  h.exam);
        set("meta-abvv-papertype", h.paper_type || h.scheme);
        set("meta-abvv-note",      h.note);
        // Also show SI paper code prominently in the QP code slot
        if (!h.qp_code && h.paper_code) set("meta-qp-code", h.paper_code);
        document.getElementById("abvv-extra-row").classList.remove("hidden");
    }
}

// ============================================================
// MARKDOWN RENDERING HELPER
// ============================================================

function renderMarkdown(text) {
    if (typeof marked === "undefined") {
        return escapeHtml(text);
    }
    return marked.parse(text || "");
}

function retypeset(element) {
    if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetPromise([element]).catch(err => console.warn("MathJax error:", err));
    }
}

// ============================================================
// HIERARCHICAL QUESTION TREE RENDERER
// ============================================================

function renderTree(tree) {
    const container = document.getElementById("questions-tree-container");
    container.innerHTML = "";

    let totalCount = 0;

    function walk(nodeMap, depth) {
        Object.entries(nodeMap).forEach(([key, node]) => {
            const nodeType = node.type || "descriptive";

            // Skip MCQ option nodes — they are rendered inside their parent MCQ card
            if (nodeType === "mcq_option") return;

            totalCount++;
            const levelClass = `level-${Math.min(depth, 4)}`;
            const card = document.createElement("div");
            card.className = `question-tome-node ${levelClass}`;
            if (nodeType === "mcq") card.classList.add("mcq-node");

            // Strip leading key from the text if it appears at the very start
            let displayText = (node.text || "").trim();
            const keyPattern = new RegExp(`^(?:Q\\.?\\s*)?${escapeRE(key)}[.\\)\\s]+`, "i");
            displayText = displayText.replace(keyPattern, "").trim();

            const marksHtml = node.marks != null && node.marks !== ""
                ? `<span class="node-marks">${node.marks}M</span>` : "";

            // Type badge
            let typeBadgeHtml = "";
            if (nodeType === "mcq") {
                typeBadgeHtml = `<span class="node-type-badge badge-mcq">MCQ</span>`;
            } else if (nodeType === "short") {
                typeBadgeHtml = `<span class="node-type-badge badge-short">Short</span>`;
            }

            // Use marked.js to render question text as markdown
            const renderedText = renderMarkdown(displayText);

            // Determine if this is a LEAF MCQ (has direct A/B/C/D options)
            // vs a CONTAINER MCQ (like Q1 that holds sub-questions i, ii, iii...)
            const hasMcqOptions = nodeType === "mcq" && node.subs &&
                Object.values(node.subs).some(s => s.type === "mcq_option");
            const hasSubQuestions = node.subs &&
                Object.values(node.subs).some(s => s.type !== "mcq_option");

            // MCQ options rendered as choice bubbles — ONLY for leaf MCQs
            let optionsHtml = "";
            if (hasMcqOptions) {
                const optionItems = Object.entries(node.subs)
                    .filter(([, ov]) => ov.type === "mcq_option")
                    .map(([ol, ov]) =>
                        `<div class="mcq-option-item">
                            <span class="mcq-option-letter">${escapeHtml(ol)}</span>
                            <span class="mcq-option-text">${escapeHtml(ov.text || "")}</span>
                        </div>`
                    ).join("");
                if (optionItems) {
                    optionsHtml = `<div class="mcq-options-grid">${optionItems}</div>`;
                }
            }

            // Extra Info toggle (only for leaf/sub nodes, not top-level Q headers)
            const canShowSpark = depth > 1 && nodeType !== "mcq_option";
            const extraInfoHtml = canShowSpark ? `
                <div class="extra-info-toggle-row" data-card-id="${escapeHtml(key)}-${depth}">
                    <span class="extra-info-label">Extra Info</span>
                    <div class="toggle-switch" role="switch" aria-checked="false" tabindex="0" id="toggle-${escapeHtml(key)}-${depth}">
                        <div class="toggle-knob"></div>
                    </div>
                    <span class="toggle-status-text" id="toggle-text-${escapeHtml(key)}-${depth}">Off</span>
                </div>` : "";

            card.innerHTML = `
                <div class="node-content-row">
                    <span class="node-key-label">${escapeHtml(key)}</span>
                    <div class="node-question-text markdown-body">${renderedText}${optionsHtml}</div>
                    <div class="node-right-meta">${typeBadgeHtml}${marksHtml}</div>
                </div>
                ${extraInfoHtml}`;

            // Typeset any math in this question card
            retypeset(card);

            // Wire up Extra Info toggle
            if (canShowSpark) {
                const toggleEl = card.querySelector(`#toggle-${CSS.escape(key)}-${depth}`);
                const toggleText = card.querySelector(`#toggle-text-${CSS.escape(key)}-${depth}`);

                if (toggleEl) {
                    const activateSpark = () => {
                        const isOn = toggleEl.getAttribute("aria-checked") === "true";
                        const newState = !isOn;
                        toggleEl.setAttribute("aria-checked", String(newState));
                        toggleEl.classList.toggle("on", newState);
                        if (toggleText) toggleText.textContent = newState ? "On" : "Off";

                        // Remove existing spark wrapper if present
                        const existing = card.querySelector(".inline-sparks-wrapper");
                        if (existing) existing.remove();

                        if (newState) {
                            // Close other open sparks in the tree
                            document.querySelectorAll(".question-tome-node").forEach(n => {
                                if (n !== card) {
                                    const otherToggle = n.querySelector(".toggle-switch");
                                    const otherText = n.querySelector(".toggle-status-text");
                                    const otherWrapper = n.querySelector(".inline-sparks-wrapper");
                                    if (otherToggle) {
                                        otherToggle.setAttribute("aria-checked", "false");
                                        otherToggle.classList.remove("on");
                                    }
                                    if (otherText) otherText.textContent = "Off";
                                    if (otherWrapper) otherWrapper.remove();
                                }
                            });

                            card.classList.add("selected");
                            const wrapper = document.createElement("div");
                            wrapper.className = "inline-sparks-wrapper";
                            card.appendChild(wrapper);

                            // Show diagram visuals if question mentions them
                            const hasVisuals = /diagram|circuit|figure|draw|sketch|table|waveform|flowchart|schematic|network|gate|karnaugh|k-map|truth/i
                                .test(node.text || displayText);
                            if (hasVisuals && activeDiagrams.length > 0) {
                                showDiagramsInWrapper(wrapper);
                            }

                            triggerInlineSpark(node.text || displayText, wrapper);
                        } else {
                            card.classList.remove("selected");
                        }
                    };

                    toggleEl.addEventListener("click", e => { e.stopPropagation(); activateSpark(); });
                    toggleEl.addEventListener("keydown", e => {
                        if (e.key === " " || e.key === "Enter") { e.preventDefault(); activateSpark(); }
                    });
                }
            }

            container.appendChild(card);

            // Recurse into subs:
            // - Always recurse for non-MCQ nodes
            // - Recurse for CONTAINER MCQs (sub-questions like i, ii, iii)
            // - SKIP recursion only for LEAF MCQs (direct A/B/C/D options already rendered as bubbles)
            const shouldRecurse = node.subs && Object.keys(node.subs).length > 0 && !hasMcqOptions;
            if (shouldRecurse) {
                walk(node.subs, depth + 1);
            }
        });
    }

    walk(tree, 1);

    // Update question count badge
    const badge = document.getElementById("q-count-badge");
    if (badge) badge.textContent = `${Object.keys(tree).length} Questions · ${totalCount} Total Nodes`;
}

function escapeRE(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function escapeHtml(str) {
    return String(str || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ============================================================
// DIAGRAM VISUALS PANEL
// ============================================================

function showDiagramsInWrapper(wrapper) {
    const strip = document.createElement("div");
    strip.className = "diagram-visuals-strip";

    const label = document.createElement("div");
    label.className = "diagram-visuals-label";
    label.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:0.85rem;height:0.85rem;flex-shrink:0">
        <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>
    </svg> Extracted Visuals from Paper`;
    strip.appendChild(label);

    const thumbsRow = document.createElement("div");
    thumbsRow.className = "diagram-thumbs-row";

    activeDiagrams.forEach((filename, idx) => {
        const url = `/api/media/${activeTaskId}/${filename}`;
        const thumb = document.createElement("div");
        thumb.className = "diagram-thumb";

        const img = document.createElement("img");
        img.src = url;
        img.alt = `Visual ${idx + 1}`;
        img.loading = "lazy";
        img.addEventListener("click", () => openDiagramLightbox(url, idx + 1));

        thumb.appendChild(img);
        thumbsRow.appendChild(thumb);
    });

    strip.appendChild(thumbsRow);
    wrapper.appendChild(strip);
}

function openDiagramLightbox(url, idx) {
    // Remove existing lightbox
    document.getElementById("diagram-lightbox")?.remove();

    const lb = document.createElement("div");
    lb.id = "diagram-lightbox";
    lb.className = "diagram-lightbox";
    lb.innerHTML = `
        <div class="lightbox-backdrop" id="lb-backdrop"></div>
        <div class="lightbox-box">
            <div class="lightbox-header">
                <span>Visual ${idx}</span>
                <button class="lightbox-close" id="lb-close">✕</button>
            </div>
            <img src="${url}" alt="Visual ${idx}" class="lightbox-img">
        </div>`;
    document.body.appendChild(lb);

    document.getElementById("lb-backdrop").addEventListener("click", () => lb.remove());
    document.getElementById("lb-close").addEventListener("click", () => lb.remove());
}

// ============================================================
// AI CURIOSITY SPARKER (INLINE NESTED ACCORDION)
// ============================================================

function triggerInlineSpark(questionText, wrapper) {
    // Show refined inline shimmer loader
    wrapper.innerHTML = `
        <div class="spark-loading-inline">
            <div class="shimmer-line title"></div>
            <div class="shimmer-line b1"></div>
            <div class="shimmer-line b2"></div>
            <div class="shimmer-line b3"></div>
        </div>`;

    fetch("/api/spark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_text: questionText })
    })
    .then(r => r.json())
    .then(data => renderInlineSpark(wrapper, data.spark))
    .catch(err => {
        wrapper.innerHTML = `<div class="spark-error-inline">Portal error: ${escapeHtml(err.message)}</div>`;
    });
}

function renderInlineSpark(wrapper, rawContent) {
    const extract = (raw, pattern) => {
        const m = raw.match(pattern);
        return m ? m[1].trim() : "";
    };

    const rw  = extract(rawContent, /\*\*Real-world Connection\*\*[:\s]+([\s\S]*?)(?=\*\*Mind-blowing Fact\*\*|$)/i);
    const mbf = extract(rawContent, /\*\*Mind-blowing Fact\*\*[:\s]+([\s\S]*?)(?=\*\*Curious Quest\*\*|$)/i);
    const cq  = extract(rawContent, /\*\*Curious Quest\*\*[:\s]+([\s\S]*?)$/i);

    // Build a single unified overview card with 3 horizontal rows
    wrapper.innerHTML = `
        <div class="inline-sparks-grid">
            <div class="spark-card-inline">

                <div class="spark-section-row spark-row-realworld">
                    <div class="spark-section-label">
                        <svg class="spark-section-label-icon" viewBox="0 0 24 24" fill="none" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="2" y1="12" x2="22" y2="12"/>
                            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
                        </svg>
                        <span class="spark-section-label-text">Real-World Link</span>
                    </div>
                    <div class="spark-section-body markdown-body">${renderMarkdown(rw || "Discovering real-world connections...")}</div>
                </div>

                <div class="spark-section-row spark-row-fact">
                    <div class="spark-section-label">
                        <svg class="spark-section-label-icon" viewBox="0 0 24 24" fill="none" stroke-width="2">
                            <polygon points="12 2 2 22 22 22 12 2"/>
                            <line x1="12" y1="9" x2="12" y2="13"/>
                            <line x1="12" y1="17" x2="12.01" y2="17"/>
                        </svg>
                        <span class="spark-section-label-text">Mind-Blowing Fact</span>
                    </div>
                    <div class="spark-section-body markdown-body">${renderMarkdown(mbf || "Surfacing fascinating academic insights...")}</div>
                </div>

                <div class="spark-section-row spark-row-quest">
                    <div class="spark-section-label">
                        <svg class="spark-section-label-icon" viewBox="0 0 24 24" fill="none" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="12" y1="16" x2="12" y2="12"/>
                            <line x1="12" y1="8" x2="12.01" y2="8"/>
                        </svg>
                        <span class="spark-section-label-text">Curious Quest</span>
                    </div>
                    <div class="spark-section-body markdown-body">${renderMarkdown(cq || "Contemplating an open intellectual challenge...")}</div>
                </div>

            </div>
        </div>`;

    retypeset(wrapper);
}

// ============================================================
// DOWNLOAD BUTTON
// ============================================================

function initDownloadButton() {
    document.getElementById("download-excel-btn").addEventListener("click", () => {
        if (activeTaskId) window.location.href = `/api/download/${activeTaskId}`;
    });
}

// ============================================================
// AI MODEL ANSWERS — GENERATE + POLL + DOWNLOAD
// ============================================================

function initAnswersButtons() {
    const generateBtn    = document.getElementById("generate-answers-btn");
    const downloadBtn    = document.getElementById("download-answers-btn");
    const statusPanel    = document.getElementById("answers-status-panel");
    const statusText     = document.getElementById("answers-status-text");
    const progressBar    = document.getElementById("answers-progress-bar");

    generateBtn.addEventListener("click", () => {
        if (!activeTaskId) {
            alert("Please upload and process a question paper first.");
            return;
        }

        // Disable generate button and show status panel
        generateBtn.disabled = true;
        generateBtn.textContent = "Generating...";
        statusPanel.classList.remove("hidden");
        downloadBtn.classList.add("hidden");
        statusText.textContent = "Dispatching 6 Groq agents in parallel...";
        progressBar.style.width = "3%";

        fetch(`/api/answers/${activeTaskId}`, { method: "POST" })
            .then(r => r.json())
            .then(() => {
                pollAnswersStatus(activeTaskId);
            })
            .catch(err => {
                statusText.textContent = `Error: ${err.message}`;
                generateBtn.disabled = false;
                generateBtn.textContent = "Generate AI Model Answer Sheet";
            });
    });

    downloadBtn.addEventListener("click", () => {
        if (activeTaskId) window.location.href = `/api/download-answers/${activeTaskId}`;
    });
}

function pollAnswersStatus(taskId) {
    const statusText  = document.getElementById("answers-status-text");
    const progressBar = document.getElementById("answers-progress-bar");
    const downloadBtn = document.getElementById("download-answers-btn");
    const generateBtn = document.getElementById("generate-answers-btn");

    answersPollInt = setInterval(() => {
        fetch(`/api/answers/status/${taskId}`)
            .then(r => r.json())
            .then(data => {
                const prog = Math.max(3, data.progress || 3);
                progressBar.style.width = `${prog}%`;
                statusText.textContent = data.message || "Processing...";

                if (data.status === "completed") {
                    clearInterval(answersPollInt);
                    progressBar.style.width = "100%";
                    statusText.textContent = "Model answer sheet compiled successfully!";
                    downloadBtn.classList.remove("hidden");
                    generateBtn.textContent = "Re-generate Answer Sheet";
                    generateBtn.disabled = false;
                } else if (data.status === "failed") {
                    clearInterval(answersPollInt);
                    statusText.textContent = `Generation failed: ${data.error || "Unknown error"}`;
                    generateBtn.disabled = false;
                    generateBtn.textContent = "Retry Generation";
                }
            })
            .catch(err => console.error("Answers polling error:", err));
    }, 1500);
}
