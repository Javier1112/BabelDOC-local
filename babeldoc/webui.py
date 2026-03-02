import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import Response
from pydantic import BaseModel

from babeldoc.format.pdf.high_level import async_translate
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.translator.translator import OpenAITranslator
from babeldoc.translator.translator import set_translate_min_interval_override
from babeldoc.translator.translator import set_translate_rate_limiter

logger = logging.getLogger(__name__)

DATA_DIR = Path.cwd() / ".babeldoc_webui"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
WORK_DIR = DATA_DIR / "work"
SETTINGS_FILE = DATA_DIR / "settings.json"
ZHIPU_DOMAINS = ("bigmodel.cn", "open.bigmodel.cn")
ZHIPU_SAFE_MAX_QPS = 2
ZHIPU_MIN_INTERVAL_SECONDS = 2.0


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_data_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)


def is_zhipu_base_url(base_url: str) -> bool:
    normalized = (base_url or "").lower()
    return any(domain in normalized for domain in ZHIPU_DOMAINS)


def get_effective_qps(settings: "AppSettings") -> int:
    qps = max(1, int(settings.qps))
    if is_zhipu_base_url(settings.openai_base_url):
        return min(qps, ZHIPU_SAFE_MAX_QPS)
    return qps


class AppSettings(BaseModel):
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    lang_in: str = "en"
    lang_out: str = "zh"
    qps: int = 4
    auto_extract_glossary: bool = False
    pool_max_workers: int = 1
    term_pool_max_workers: int = 1
    skip_reference_section: bool = True


@dataclass
class JobState:
    job_id: str
    file_name: str
    file_path: str
    status: str = "queued"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    stage: str = ""
    stage_progress: float = 0.0
    stage_current: int = 0
    stage_total: int = 0
    overall_progress: float = 0.0
    error: str | None = None
    mono_pdf_path: str | None = None
    dual_pdf_path: str | None = None
    no_watermark_mono_pdf_path: str | None = None
    no_watermark_dual_pdf_path: str | None = None


class AppState:
    def __init__(self):
        self.jobs: dict[str, JobState] = {}
        self._doc_layout_model = None
        self._doc_layout_lock = asyncio.Lock()
        ensure_data_dirs()

    async def get_doc_layout_model(self):
        if self._doc_layout_model is not None:
            return self._doc_layout_model
        async with self._doc_layout_lock:
            if self._doc_layout_model is None:
                from babeldoc.docvision.doclayout import DocLayoutModel

                self._doc_layout_model = await asyncio.to_thread(
                    DocLayoutModel.load_onnx
                )
        return self._doc_layout_model


state = AppState()
app = FastAPI(title="BabelDOC Local WebUI")


def load_settings() -> AppSettings:
    ensure_data_dirs()
    if not SETTINGS_FILE.exists():
        default_settings = AppSettings()
        save_settings(default_settings)
        return default_settings
    raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    return AppSettings(**raw)


def save_settings(settings: AppSettings) -> None:
    ensure_data_dirs()
    SETTINGS_FILE.write_text(
        settings.model_dump_json(indent=2),
        encoding="utf-8",
    )


def serialize_job(job: JobState) -> dict[str, Any]:
    return asdict(job)


async def run_translation_job(
    job_id: str,
    pages: str | None,
    no_dual: bool,
    no_mono: bool,
    output_dir: str | None = None,
) -> None:
    job = state.jobs[job_id]
    settings = load_settings()
    if not settings.openai_api_key:
        job.status = "failed"
        job.error = "API Key is empty. Set it in settings first."
        job.updated_at = utc_now_iso()
        return

    try:
        job.status = "running"
        job.updated_at = utc_now_iso()

        translator = OpenAITranslator(
            lang_in=settings.lang_in,
            lang_out=settings.lang_out,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            ignore_cache=False,
        )
        effective_qps = get_effective_qps(settings)
        effective_pool_workers = max(1, int(settings.pool_max_workers))
        effective_term_pool_workers = max(1, int(settings.term_pool_max_workers))
        set_translate_rate_limiter(effective_qps)
        if is_zhipu_base_url(settings.openai_base_url):
            set_translate_min_interval_override(ZHIPU_MIN_INTERVAL_SECONDS)
        else:
            set_translate_min_interval_override(None)
        doc_layout_model = await state.get_doc_layout_model()

        base_output_dir = OUTPUT_DIR
        if output_dir and output_dir.strip():
            base_output_dir = Path(output_dir.strip()).expanduser()
        job_output_dir = base_output_dir / job_id
        job_work_dir = WORK_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        job_work_dir.mkdir(parents=True, exist_ok=True)

        config = TranslationConfig(
            translator=translator,
            input_file=job.file_path,
            lang_in=settings.lang_in,
            lang_out=settings.lang_out,
            doc_layout_model=doc_layout_model,
            output_dir=job_output_dir,
            working_dir=job_work_dir,
            pages=pages,
            qps=effective_qps,
            no_dual=no_dual,
            no_mono=no_mono,
            report_interval=0.5,
            auto_extract_glossary=bool(settings.auto_extract_glossary),
            pool_max_workers=effective_pool_workers,
            term_pool_max_workers=effective_term_pool_workers,
            skip_reference_section=bool(settings.skip_reference_section),
        )

        async for event in async_translate(config):
            event_type = event.get("type")
            if event_type in {"progress_start", "progress_update", "progress_end"}:
                job.stage = str(event.get("stage", ""))
                job.stage_progress = float(event.get("stage_progress", 0.0))
                job.stage_current = int(event.get("stage_current", 0))
                job.stage_total = int(event.get("stage_total", 0))
                job.overall_progress = float(event.get("overall_progress", 0.0))
                job.updated_at = utc_now_iso()
            elif event_type == "finish":
                result = event.get("translate_result")
                if result is not None:
                    job.mono_pdf_path = str(result.mono_pdf_path) if result.mono_pdf_path else None
                    job.dual_pdf_path = str(result.dual_pdf_path) if result.dual_pdf_path else None
                    job.no_watermark_mono_pdf_path = (
                        str(result.no_watermark_mono_pdf_path)
                        if result.no_watermark_mono_pdf_path
                        else None
                    )
                    job.no_watermark_dual_pdf_path = (
                        str(result.no_watermark_dual_pdf_path)
                        if result.no_watermark_dual_pdf_path
                        else None
                    )
                job.status = "success"
                job.overall_progress = 100.0
                job.updated_at = utc_now_iso()
            elif event_type == "error":
                job.status = "failed"
                job.error = str(event.get("error", "Unknown error"))
                job.updated_at = utc_now_iso()
                break

        if job.status == "running":
            job.status = "success"
            job.overall_progress = 100.0
            job.updated_at = utc_now_iso()
    except Exception as e:
        logger.exception("Translation job failed")
        job.status = "failed"
        job.error = str(e)
        job.updated_at = utc_now_iso()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=INDEX_HTML)


@app.get("/api/settings")
async def get_settings():
    return load_settings().model_dump()


@app.post("/api/settings")
async def update_settings(settings: AppSettings):
    if settings.qps <= 0:
        raise HTTPException(status_code=400, detail="qps must be > 0")
    if settings.pool_max_workers <= 0:
        raise HTTPException(status_code=400, detail="pool_max_workers must be > 0")
    if settings.term_pool_max_workers <= 0:
        raise HTTPException(
            status_code=400, detail="term_pool_max_workers must be > 0"
        )
    if is_zhipu_base_url(settings.openai_base_url) and settings.qps > ZHIPU_SAFE_MAX_QPS:
        settings.qps = ZHIPU_SAFE_MAX_QPS
    save_settings(settings)
    return {"ok": True}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    pages: str | None = Form(default=None),
    no_dual: bool = Form(default=False),
    no_mono: bool = Form(default=False),
    output_dir: str | None = Form(default=None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file name is empty")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDF is supported")

    job_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name
    save_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    file_bytes = await file.read()
    save_path.write_bytes(file_bytes)

    job = JobState(
        job_id=job_id,
        file_name=safe_name,
        file_path=str(save_path),
    )
    state.jobs[job_id] = job

    asyncio.create_task(
        run_translation_job(job_id, pages, no_dual, no_mono, output_dir)
    )
    return {"job_id": job_id}


@app.get("/api/jobs")
async def list_jobs():
    jobs = [serialize_job(job) for job in state.jobs.values()]
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return serialize_job(job)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


def cli() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    host = os.getenv("BABELDOC_WEBUI_HOST", "127.0.0.1")
    port = int(os.getenv("BABELDOC_WEBUI_PORT", "7861"))
    uvicorn.run("babeldoc.webui:app", host=host, port=port, reload=False)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>BabelDOC Local WebUI</title>
  <style>
    :root { --bg: #f5f7fb; --card: #ffffff; --line: #d7deea; --text: #172035; --muted: #5f6b85; --brand: #1a73e8; }
    body { margin: 0; font-family: "Segoe UI", "PingFang SC", sans-serif; background: var(--bg); color: var(--text); }
    .wrap { max-width: 960px; margin: 20px auto; padding: 0 12px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin-bottom: 12px; }
    h1 { font-size: 22px; margin: 0 0 12px; }
    h2 { font-size: 16px; margin: 0 0 10px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    label { font-size: 13px; color: var(--muted); display: block; margin-bottom: 4px; }
    input[type=text], input[type=number] { width: 100%; padding: 8px; border: 1px solid var(--line); border-radius: 8px; box-sizing: border-box; }
    button { border: 0; background: var(--brand); color: white; padding: 8px 12px; border-radius: 8px; cursor: pointer; }
    .muted { color: var(--muted); font-size: 12px; }
    .row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .task { border: 1px solid var(--line); border-radius: 8px; padding: 10px; margin-bottom: 8px; }
    .status { font-weight: 600; }
    .bar { height: 8px; background: #e9edf6; border-radius: 100px; overflow: hidden; margin: 6px 0; }
    .bar > span { display: block; height: 100%; background: #1a73e8; }
    .err { color: #ba1a1a; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>BabelDOC Local WebUI</h1>

    <div class="card">
      <h2>API Settings</h2>
      <div class="grid">
        <div><label>API Key</label><input id="api_key" type="text" /></div>
        <div><label>Base URL</label><input id="base_url" type="text" /></div>
        <div><label>Model</label><input id="model" type="text" /></div>
        <div><label>QPS</label><input id="qps" type="number" min="1" /></div>
        <div><label>Auto Extract Glossary</label><input id="auto_extract_glossary" type="checkbox" /></div>
        <div><label>Pool Workers</label><input id="pool_max_workers" type="number" min="1" /></div>
        <div><label>Term Pool Workers</label><input id="term_pool_max_workers" type="number" min="1" /></div>
        <div><label>Skip Reference Section</label><input id="skip_reference_section" type="checkbox" /></div>
        <div><label>Source Language</label><input id="lang_in" type="text" /></div>
        <div><label>Target Language</label><input id="lang_out" type="text" /></div>
      </div>
      <div class="row" style="margin-top:10px;">
        <button id="save_settings">Save Settings</button>
        <span class="muted" id="settings_msg"></span>
      </div>
    </div>

    <div class="card">
      <h2>Create Translation Job</h2>
      <div class="grid">
        <div><label>PDF File</label><input id="pdf_file" type="file" accept=".pdf" /></div>
        <div><label>Pages (Optional, e.g. 1-3,6)</label><input id="pages" type="text" /></div>
        <div><label>Output Folder (Optional)</label><input id="output_dir" type="text" placeholder="e.g. F:/translated_output" /></div>
      </div>
      <div class="row" style="margin-top:10px;">
        <label><input id="no_dual" type="checkbox" /> Disable bilingual PDF</label>
        <label><input id="no_mono" type="checkbox" /> Disable monolingual PDF</label>
      </div>
      <div class="row" style="margin-top:10px;">
        <button id="create_job">Start Translation</button>
        <span class="muted" id="job_msg"></span>
      </div>
    </div>

    <div class="card">
      <h2>Job Progress</h2>
      <div id="jobs"></div>
    </div>
  </div>

  <script>
    async function loadSettings() {
      const res = await fetch('/api/settings');
      const s = await res.json();
      document.getElementById('api_key').value = s.openai_api_key || '';
      document.getElementById('base_url').value = s.openai_base_url || '';
      document.getElementById('model').value = s.openai_model || '';
      document.getElementById('qps').value = s.qps || 4;
      document.getElementById('auto_extract_glossary').checked = Boolean(s.auto_extract_glossary);
      document.getElementById('pool_max_workers').value = s.pool_max_workers || 1;
      document.getElementById('term_pool_max_workers').value = s.term_pool_max_workers || 1;
      document.getElementById('skip_reference_section').checked = s.skip_reference_section !== false;
      document.getElementById('lang_in').value = s.lang_in || 'en';
      document.getElementById('lang_out').value = s.lang_out || 'zh';
    }

    async function saveSettings() {
      const payload = {
        openai_api_key: document.getElementById('api_key').value.trim(),
        openai_base_url: document.getElementById('base_url').value.trim(),
        openai_model: document.getElementById('model').value.trim(),
        qps: Number(document.getElementById('qps').value || 4),
        auto_extract_glossary: document.getElementById('auto_extract_glossary').checked,
        pool_max_workers: Number(document.getElementById('pool_max_workers').value || 1),
        term_pool_max_workers: Number(document.getElementById('term_pool_max_workers').value || 1),
        skip_reference_section: document.getElementById('skip_reference_section').checked,
        lang_in: document.getElementById('lang_in').value.trim() || 'en',
        lang_out: document.getElementById('lang_out').value.trim() || 'zh'
      };
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const msg = document.getElementById('settings_msg');
      msg.textContent = res.ok ? 'Settings saved.' : 'Save failed.';
    }

    async function createJob() {
      const fileInput = document.getElementById('pdf_file');
      const file = fileInput.files[0];
      const msg = document.getElementById('job_msg');
      if (!file) { msg.textContent = 'Please choose a PDF file first.'; return; }
      const fd = new FormData();
      fd.append('file', file);
      fd.append('pages', document.getElementById('pages').value.trim());
      fd.append('output_dir', document.getElementById('output_dir').value.trim());
      fd.append('no_dual', document.getElementById('no_dual').checked ? 'true' : 'false');
      fd.append('no_mono', document.getElementById('no_mono').checked ? 'true' : 'false');
      const res = await fetch('/api/jobs', { method: 'POST', body: fd });
      if (res.ok) {
        const data = await res.json();
        msg.textContent = 'Job created: ' + data.job_id;
        fileInput.value = '';
      } else {
        let err = 'Create failed';
        try { const data = await res.json(); err = data.detail || err; } catch(e) {}
        msg.textContent = err;
      }
    }

    function renderJobs(items) {
      const container = document.getElementById('jobs');
      if (!items.length) {
        container.innerHTML = '<div class="muted">No jobs yet.</div>';
        return;
      }
      container.innerHTML = items.map(j => {
        const p = Number(j.overall_progress || 0).toFixed(1);
        const output = [j.mono_pdf_path, j.dual_pdf_path].filter(Boolean).join(' | ');
        return `
          <div class="task">
            <div><strong>${j.file_name}</strong></div>
            <div class="status">Status: ${j.status}</div>
            <div class="bar"><span style="width:${Math.max(0, Math.min(100, p))}%"></span></div>
            <div class="muted">Progress: ${p}% | Stage: ${j.stage || '-'} (${j.stage_current}/${j.stage_total})</div>
            ${j.error ? `<div class="err">Error: ${j.error}</div>` : ''}
            ${output ? `<div class="muted">Output: ${output}</div>` : ''}
          </div>
        `;
      }).join('');
    }

    let loadJobsInFlight = false;
    async function loadJobs() {
      if (loadJobsInFlight) return;
      loadJobsInFlight = true;
      try {
        const res = await fetch('/api/jobs');
        if (!res.ok) return;
        const data = await res.json();
        renderJobs(data.jobs || []);
      } finally {
        loadJobsInFlight = false;
      }
    }

    document.getElementById('save_settings').addEventListener('click', saveSettings);
    document.getElementById('create_job').addEventListener('click', createJob);

    loadSettings();
    loadJobs();
    setInterval(loadJobs, 3000);
  </script>
</body>
</html>
"""