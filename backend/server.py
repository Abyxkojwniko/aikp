# -*- coding: utf-8 -*-
import sys as _sys
# Windows consoles default to GBK; a player pasting an emoji/CJK-ext char would
# otherwise crash any print() with UnicodeEncodeError → HTTP 500. Make all logging
# output encoding-proof.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import time, uuid, json, os as _os
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional
from engine import run_gm_turn
from config import WORLD_BOOK_DIR, SESSIONS_DIR, BACKEND_DIR
from state_manager import load_session

app = FastAPI(title="AIKP GM Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOADS_DIR = _os.path.join(BACKEND_DIR, "uploads")
_os.makedirs(UPLOADS_DIR, exist_ok=True)

import threading
_parse_jobs: dict[str, dict] = {}


def _feed_to_rag(world: dict) -> None:
    """Index world book entities into ChromaDB per-module collection."""
    try:
        from rag import index_world_book
        name = world.get("name", "unknown")
        count = index_world_book(name, world)
        print(f"[AIKP] RAG: indexed {count} entities for '{name}'", flush=True)
    except Exception as e:
        print(f"[AIKP] RAG index error: {e}", flush=True)


def _read_st_secret() -> str:
    """Read API key from SillyTavern secrets.json as fallback."""
    try:
        st_secrets = _os.path.normpath(_os.path.join(
            BACKEND_DIR, "..", "Tavern", "SillyTavern",
            "data", "default-user", "secrets.json"
        ))
        print(f"[AIKP] Reading secrets from: {st_secrets}", flush=True)
        print(f"[AIKP]   exists: {_os.path.exists(st_secrets)}", flush=True)
        if _os.path.exists(st_secrets):
            with open(st_secrets, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("api_key_custom", [])
            print(f"[AIKP]   entries found: {len(entries)}", flush=True)
            for entry in entries:
                key = entry.get("value", "")
                if key:
                    print(f"[AIKP]   using key: {key[:10]}...", flush=True)
                    return key
        print("[AIKP]   no key found in secrets", flush=True)
    except Exception as e:
        import traceback
        print(f"[AIKP] Read secrets error: {e}", flush=True)
        traceback.print_exc()
    return ""


def _resolve_api_key(fast_request, use_secret: bool = True) -> str:
    """Resolve API key: header → config.py → SillyTavern secrets."""
    auth_header = fast_request.headers.get("Authorization", "")
    api_key = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    from config import DEEPSEEK_API_KEY
    if not api_key:
        api_key = DEEPSEEK_API_KEY
    if not api_key and use_secret:
        api_key = _read_st_secret()
    return api_key


class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "aikp-gm"
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

class ChatMessage(BaseModel):
    role: str = "assistant"
    content: str

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Usage()


@app.get("/v1/models")
def list_models():
    models = []
    if _os.path.isdir(WORLD_BOOK_DIR):
        for entry in sorted(_os.listdir(WORLD_BOOK_DIR)):
            epath = _os.path.join(WORLD_BOOK_DIR, entry)
            if _os.path.isdir(epath):
                # Look for {name}.json inside the folder
                for f in _os.listdir(epath):
                    if f.endswith(".json"):
                        name = f[:-5]
                        models.append({"id": name, "object": "model", "owned_by": "aikp"})
                        break
            elif entry.endswith(".json"):
                name = entry[:-5]
                models.append({"id": name, "object": "model", "owned_by": "aikp"})
    if not models:
        models.append({"id": "tavern_trial", "object": "model", "owned_by": "aikp"})
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, fast_request: Request):
    # DEBUG: log everything
    print("=" * 50)
    print("[AIKP] REQUEST RECEIVED")
    print("[AIKP]   model:", req.model)
    print("[AIKP]   stream:", req.stream)
    print("[AIKP]   messages:", len(req.messages))
    print("[AIKP]   headers:", dict(fast_request.headers))

    chat_id = f"{req.model}-session"
    auth_header = fast_request.headers.get("Authorization", "")
    api_key = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    print("[AIKP]   auth header:", "present" if auth_header else "MISSING")
    print("[AIKP]   key length:", len(api_key))

    messages = [m.model_dump() for m in req.messages]
    print(f"[AIKP]   message breakdown ({len(messages)} total):", flush=True)
    for _i, _m in enumerate(messages):
        _c = (_m.get("content") or "")
        print(f"[AIKP]     [{_i}] role={_m.get('role')!r} len={len(_c)} content={_c[:150]!r}", flush=True)
    print("[AIKP]   last user msg:", messages[-1]["content"][:80] if messages else "NONE")

    try:
        result = run_gm_turn(messages=messages, model=req.model, chat_id=chat_id, api_key=api_key, stream=req.stream)
    except Exception as e:
        print(f"[AIKP] ENGINE ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    print(f"[AIKP] result type: {type(result).__name__}", flush=True)
    print(f"[AIKP] iterable: {hasattr(result, '__iter__')}", flush=True)

    if req.stream and hasattr(result, "__iter__"):
        print("[AIKP] => Returning SSE stream", flush=True)
        async def generate():
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            n = chr(10)
            full_text = ""
            try:
                for chunk in result:
                    content = ""
                    try:
                        c = chunk.choices[0]
                        if c.delta and c.delta.content:
                            content = c.delta.content
                    except Exception:
                        pass
                    if content:
                        full_text += content
                        d = json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model, "choices": [{"index": 0, "delta": {"content": content}}]})
                        yield f"data: {d}{n}{n}"
                end = json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                yield f"data: {end}{n}{n}"
                yield f"data: [DONE]{n}{n}"
                # Backfill the actual GM response into the turn log for streaming
                if full_text:
                    try:
                        from state_manager import load_session, update_last_turn_response, save_session
                        sess = load_session(chat_id)
                        update_last_turn_response(sess, full_text)
                        save_session(sess)
                    except Exception as be:
                        print(f"[AIKP] Failed to backfill turn log: {be}", flush=True)
                print("[AIKP] SSE stream complete", flush=True)
            except Exception as e:
                print(f"[AIKP] SSE error: {e}", flush=True)
                import traceback; traceback.print_exc()
                err = json.dumps({"error": {"message": str(e), "type": "server_error"}})
                yield f"data: {err}{n}{n}"
                yield f"data: [DONE]{n}{n}"
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        print("[AIKP] => Returning JSON", flush=True)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=[Choice(message=ChatMessage(role="assistant", content=result))],
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/worlds")
def list_worlds():
    worlds = []
    seen = set()
    if _os.path.isdir(WORLD_BOOK_DIR):
        for entry in sorted(_os.listdir(WORLD_BOOK_DIR)):
            epath = _os.path.join(WORLD_BOOK_DIR, entry)
            # Subdirectory format: models/{name}/{name}.json
            if _os.path.isdir(epath) and not entry.startswith("_"):
                for f in _os.listdir(epath):
                    if f.endswith(".json"):
                        name = f[:-5]
                        if name in seen:
                            continue
                        seen.add(name)
                        try:
                            with open(_os.path.join(epath, f), "r", encoding="utf-8-sig") as fp:
                                data = json.load(fp)
                            worlds.append({
                                "id": name,
                                "name": data.get("name", name),
                                "description": data.get("description", ""),
                                "scene_count": len(data.get("scenes", {})),
                                "entity_count": len(data.get("entities", {})),
                            })
                        except Exception:
                            worlds.append({"id": name, "name": name, "description": ""})
            # Flat format: models/{name}.json
            elif entry.endswith(".json") and not entry.startswith("_"):
                name = entry[:-5]
                if name in seen:
                    continue
                seen.add(name)
                try:
                    with open(epath, "r", encoding="utf-8-sig") as fp:
                        data = json.load(fp)
                    worlds.append({
                        "id": name,
                        "name": data.get("name", name),
                        "description": data.get("description", ""),
                        "scene_count": len(data.get("scenes", {})),
                        "entity_count": len(data.get("entities", {})),
                    })
                except Exception:
                    worlds.append({"id": name, "name": name, "description": ""})
    return worlds


@app.delete("/api/worlds/{name}")
def delete_world(name: str):
    """Delete a world book JSON file and its ChromaDB collection.
    
    Removes: models/{name}/ directory or models/{name}.json,
    ChromaDB collection aikp_{name}, and in-memory caches.
    """
    import shutil

    # Sanitize name to prevent path traversal
    safe_name = _os.path.basename(name)
    if safe_name != name or ".." in name or name.startswith("_"):
        raise HTTPException(status_code=400, detail="Invalid world book name")

    deleted = False

    # Delete folder-style world book: models/{name}/
    folder_path = _os.path.join(WORLD_BOOK_DIR, safe_name)
    if _os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        deleted = True

    # Delete flat-style world book: models/{name}.json
    flat_path = _os.path.join(WORLD_BOOK_DIR, f"{safe_name}.json")
    if _os.path.isfile(flat_path):
        _os.remove(flat_path)
        deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail=f"World book '{safe_name}' not found")

    # Delete ChromaDB collection
    try:
        from rag import delete_collection
        delete_collection(safe_name)
    except Exception as e:
        print(f"[AIKP] Failed to delete ChromaDB collection for '{name}': {e}", flush=True)

    # Invalidate in-memory caches
    try:
        from engine import invalidate_world_cache
        invalidate_world_cache(safe_name)
    except Exception:
        pass

    return {"status": "deleted", "name": safe_name}


@app.get("/api/sessions")
def list_sessions():
    sessions = []
    if _os.path.isdir(SESSIONS_DIR):
        for f in sorted(_os.listdir(SESSIONS_DIR)):
            if f.endswith(".json"):
                chat_id = f[:-5]
                path = _os.path.join(SESSIONS_DIR, f)
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    sessions.append({
                        "chat_id": chat_id,
                        "model": data.get("model", ""),
                        "current_turn": data.get("current_turn", 0),
                        "updated_at": data.get("updated_at", 0),
                        "plot_phase": data.get("plot_phase", "intro"),
                    })
                except Exception:
                    sessions.append({"chat_id": chat_id, "error": "corrupt"})
    return sessions


@app.get("/api/session/{chat_id}")
def get_session(chat_id: str):
    try:
        session = load_session(chat_id)
        session.pop("turn_log", None)
        # Attach human-readable scene name for the frontend status panel
        try:
            from engine import load_world
            model = session.get("model", "")
            ps = session.get("player_state")
            if model and isinstance(ps, dict):
                world = load_world(model)
                sid = ps.get("current_scene", "")
                ps["current_scene_name"] = world.get("scenes", {}).get(sid, {}).get("name", sid)
        except Exception:
            pass
        return session
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/session/{chat_id}/reset")
def reset_session(chat_id: str):
    import os
    from state_manager import session_path
    path = session_path(chat_id)
    if path.exists():
        os.remove(path)
    # Also clear engine's in-memory session cache so next load reads fresh state.
    try:
        from engine import _session_cache
        _session_cache.pop(chat_id, None)
    except Exception:
        pass
    return {"status": "ok", "chat_id": chat_id}


_VERDICT_CN = {
    "critical_success": "大成功", "extreme_success": "极难成功",
    "hard_success": "困难成功", "success": "成功",
    "failure": "失败", "critical_failure": "大失败", "fumble": "大失败",
}


@app.post("/api/roll/{chat_id}")
def roll_check(chat_id: str):
    """Player clicked the dice button — roll the pending check now (server-side
    RNG), grade it (CoC success levels), apply state transition + SAN, clear the
    pending check, and return the roll for the UI to show."""
    from state_manager import load_session, save_session
    from dice import resolve_check, coc_skill_check, coc_san_loss
    from engine import load_world

    session = load_session(chat_id)
    pc = session.get("pending_check")
    if not pc:
        raise HTTPException(status_code=400, detail="没有待掷的检定")

    world = load_world(session.get("model", ""))
    entity = world.get("entities", {}).get(pc.get("entity_id", ""), {})
    state_def = entity.get("states", {}).get(pc.get("state", ""), {})
    rule = pc.get("rule_system", "coc")
    ps = session.setdefault("player_state", {})

    out = {"entity": pc.get("entity_id"), "entity_name": entity.get("name", ""),
           "skill": pc.get("skill", ""), "narration": ""}
    narration = []

    # Skill check: main d100 vs target → success level
    if pc.get("skill"):
        if rule == "coc":
            r = coc_skill_check(pc.get("effective", 0))
        else:
            r = resolve_check(rule, pc.get("effective", 0), pc.get("dc", 12))
        out["check"] = {
            "skill": pc.get("skill"), "roll": r.get("d100") or r.get("d20"),
            "target": pc.get("effective", 0), "success": r.get("success"),
            "verdict": r.get("verdict"),
            "verdict_cn": _VERDICT_CN.get(r.get("verdict", ""), r.get("verdict", "")),
        }
        if pc.get("_scene_clue_id"):
            # Scene clue check: on success → reveal clue description
            clue_id = pc["_scene_clue_id"]
            scene_id = pc.get("scene", "")
            clues = world.get("scenes", {}).get(scene_id, {}).get("clues", [])
            clue = next((c for c in clues if c.get("id") == clue_id), {})
            if r.get("success"):
                session.setdefault("discovered_clues", []).append(clue_id)
                clue_desc = clue.get("desc", "")
                narration.append(f"〈{pc.get('skill')}〉检定：{out['check']['verdict_cn']}。")
                if clue_desc:
                    narration.append(f"【线索发现】{clue_desc}")
                print(f"[ENGINE] Scene clue discovered: {clue_id}", flush=True)
            else:
                narration.append(f"〈{pc.get('skill')}〉检定：{out['check']['verdict_cn']}。你未能发现什么特别的线索。")
        elif pc.get("dynamic"):
            # LLM-initiated check: no module on_pass/on_fail. Record the outcome
            # so next turn's context tells the KP to narrate what it revealed.
            session["_last_check_result"] = {
                "skill": pc.get("skill"), "success": r.get("success"),
                "verdict": r.get("verdict"), "verdict_cn": out["check"]["verdict_cn"],
            }
            narration.append(f"〈{pc.get('skill')}〉检定：{out['check']['verdict_cn']}。")
        else:
            tr = state_def.get("on_pass" if r.get("success") else "on_fail", {})
            nxt = tr.get("to_state")
            if nxt:
                session.setdefault("entity_states", {})[pc["entity_id"]] = nxt
            if tr.get("narration"):
                narration.append(tr["narration"])
        print(f"[ENGINE] Player rolled {pc['skill']}: roll={out['check']['roll']} "
              f"vs {pc.get('effective')} → {r.get('verdict')}", flush=True)

    # SAN check: d100 vs current SAN → lose success/failure dice
    if pc.get("san_check"):
        cur = ps.get("san", 0)
        sr = coc_san_loss(pc["san_check"], cur)
        ps["san"] = max(0, sr["san_after"])
        out["san"] = {
            "roll": sr["d100"], "vs_san": cur, "passed": sr["passed_pow_check"],
            "loss": sr["san_loss"], "before": cur, "after": ps["san"],
            "insanity_temp": sr["insanity_temp"], "insanity_indef": sr["insanity_indef"],
        }
        print(f"[ENGINE] Player rolled SAN: d100={sr['d100']} vs {cur} → "
              f"loss={sr['san_loss']} (now {ps['san']})", flush=True)
        # SAN-only check: transition the entity's state too, so looking once
        # doesn't keep re-triggering the same SAN loss every turn.
        tr = state_def.get("on_pass") or state_def.get("on_trigger") or {}
        nxt = tr.get("to_state")
        if nxt:
            session.setdefault("entity_states", {})[pc["entity_id"]] = nxt

    session["pending_check"] = None
    save_session(session)
    # Sync engine's in-memory session cache — otherwise the next turn's
    # run_gm_turn reuses a stale cached session and the roll's SAN/state changes
    # are lost (player would see SAN reset back).
    try:
        from engine import _session_cache
        _session_cache[chat_id] = session
    except Exception:
        pass
    out["narration"] = "\n".join(narration)
    return out


@app.post("/api/upload")
async def upload_module(file: UploadFile = File(...)):
    """Upload a TRPG module file (.txt, .md, .docx)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = _os.path.splitext(file.filename)[1].lower()
    if ext not in (".txt", ".md", ".docx", ".pdf"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Use .txt, .md, .docx, or .pdf")

    upload_id = uuid.uuid4().hex[:12]
    safe_name = f"{upload_id}{ext}"
    save_path = _os.path.join(UPLOADS_DIR, safe_name)

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    with open(save_path, "wb") as f:
        f.write(content)

    # Extract text for preview
    text = ""
    if ext == ".docx":
        text = _extract_docx_text(save_path)
    elif ext == ".pdf":
        text = _extract_pdf_text(save_path)
    else:
        text = content.decode("utf-8", errors="replace")

    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "size": len(content),
        "char_count": len(text),
        "preview": text[:500],
        "status": "uploaded",
    }


def _extract_docx_text(path: str) -> str:
    """Extract plain text from a .docx file."""
    import zipfile, xml.etree.ElementTree as ET
    try:
        z = zipfile.ZipFile(path)
        xml = z.read("word/document.xml")
        root = ET.fromstring(xml)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        lines = []
        for p in root.iter(f"{{{ns['w']}}}p"):
            line = "".join(t.text or "" for t in p.iter(f"{{{ns['w']}}}t"))
            if line.strip():
                lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        return f"[Error extracting .docx: {e}]"


def _extract_pdf_text(path: str) -> str:
    """Extract plain text from a .pdf file using pypdf.
    
    Uses pypdf (pure Python, ~200KB, zero external dependencies).
    For scanned/image-based PDFs without a text layer, this will return
    empty text. Users should use SillyTavern's client-side pdf.js path instead.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        lines = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                lines.append(text)
        result = "\n".join(lines)
        if not result.strip():
            return "[PDF contains no extractable text. If this is a scanned document, upload through SillyTavern which uses client-side pdf.js for better extraction.]"
        return result
    except ImportError:
        return "[pypdf not installed. Run: pip install pypdf]"
    except Exception as e:
        return f"[Error extracting PDF: {e}]"


@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    """Simple upload page for TRPG module files."""
    return HTMLResponse("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIKP — Upload Module</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
.card { background: #16213e; border-radius: 12px; padding: 40px; max-width: 500px; width: 90%; box-shadow: 0 8px 32px rgba(0,0,0,.4); }
h1 { font-size: 1.5em; margin-bottom: 8px; }
h1 span { color: #e94560; }
.sub { color: #888; margin-bottom: 24px; font-size: .9em; }
.dropzone { border: 2px dashed #444; border-radius: 8px; padding: 40px 20px; text-align: center; cursor: pointer; transition: .2s; margin-bottom: 16px; }
.dropzone:hover, .dropzone.drag { border-color: #e94560; background: rgba(233,69,96,.05); }
.dropzone p { color: #888; }
.dropzone .icon { font-size: 2em; margin-bottom: 8px; }
input[type=file] { display: none; }
#status { margin-top: 16px; padding: 12px; border-radius: 6px; display: none; }
#status.success { display: block; background: #0f3460; color: #4ecca3; }
#status.error { display: block; background: #3a0a0a; color: #e94560; }
#preview { margin-top: 16px; padding: 12px; background: #0a0a1a; border-radius: 6px; font-size: .85em; max-height: 200px; overflow-y: auto; white-space: pre-wrap; display: none; }
.spinner { display: none; width: 20px; height: 20px; border: 2px solid #444; border-top-color: #e94560; border-radius: 50%; animation: spin .6s linear infinite; margin: 12px auto; }
@keyframes spin { to { transform: rotate(360deg); } }
.fmts { color: #666; font-size: .8em; margin-top: 8px; }
</style>
</head>
<body>
<div class="card">
  <h1>AIKP <span>Module Upload</span></h1>
  <p class="sub">Upload a TRPG scenario file to parse and play</p>
  <div class="dropzone" id="dropzone">
    <div class="icon">📄</div>
    <p>Click or drag .txt / .md / .docx here</p>
    <p class="fmts">Max 50MB</p>
  </div>
  <input type="file" id="fileInput" accept=".txt,.md,.docx">
  <div class="spinner" id="spinner"></div>
  <div id="status"></div>
  <div id="preview"></div>
</div>
<script>
const dz = document.getElementById('dropzone');
const fi = document.getElementById('fileInput');
const st = document.getElementById('status');
const pv = document.getElementById('preview');
const sp = document.getElementById('spinner');

dz.addEventListener('click', () => fi.click());
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); if(e.dataTransfer.files.length) upload(e.dataTransfer.files[0]); });
fi.addEventListener('change', () => { if(fi.files.length) upload(fi.files[0]); });

async function upload(file) {
  st.style.display = 'none'; pv.style.display = 'none'; sp.style.display = 'block';
  const form = new FormData(); form.append('file', file);
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: form });
    const d = await r.json();
    sp.style.display = 'none';
    if (r.ok) {
      st.className = 'success'; st.style.display = 'block';
      st.textContent = `Uploaded: ${d.filename} (${(d.size/1024).toFixed(1)}KB, ${d.char_count} chars)`;
      if (d.preview) { pv.style.display = 'block'; pv.textContent = 'Preview:\\n' + d.preview; }
    } else {
      st.className = 'error'; st.style.display = 'block';
      st.textContent = 'Error: ' + (d.detail || 'Upload failed');
    }
  } catch(e) {
    sp.style.display = 'none';
    st.className = 'error'; st.style.display = 'block';
    st.textContent = 'Error: ' + e.message;
  }
}
</script>
</body>
</html>""")


class ParseTextRequest(BaseModel):
    text: str
    filename: str = "module.txt"


class ParseLocalRequest(BaseModel):
    path: str          # relative URL like "/user/files/xxx.docx"
    filename: str = "module.txt"


@app.post("/api/parse/local")
def parse_local(req: ParseLocalRequest, fast_request: Request):
    """Parse a file already stored locally by SillyTavern Data Bank.
    Reads the file from disk by resolving the path against ST's data directory."""
    st_data = _os.path.normpath(_os.path.join(
        BACKEND_DIR, "..", "Tavern", "SillyTavern", "data", "default-user"
    ))
    # req.path is like "/user/files/123_abc.bin" — the /user/ prefix is
    # a SillyTavern URL route; the actual file is at files/xxx under the user dir.
    rel = req.path.lstrip("/")
    if rel.startswith("user/"):
        rel = rel[len("user/"):]
    file_path = _os.path.normpath(_os.path.join(st_data, rel))
    if not file_path.startswith(st_data):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not _os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {rel}")

    ext = _os.path.splitext(req.filename)[1].lower()
    if ext == ".docx":
        text = _extract_docx_text(file_path)
    elif ext == ".pdf":
        text = _extract_pdf_text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    if not text or len(text) < 50:
        raise HTTPException(status_code=400, detail="File too short or unreadable")

    upload_id = uuid.uuid4().hex[:12]
    save_path = _os.path.join(UPLOADS_DIR, f"{upload_id}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    api_key = _resolve_api_key(fast_request)
    _parse_jobs[upload_id] = {"status": "running", "progress": 0, "step": "starting"}

    def _run():
        try:
            from parser import ModuleParser, save_world_book
            _parse_jobs[upload_id] = {"status": "running", "progress": 5, "step": "parsing"}
            parser_obj = ModuleParser(api_key=api_key)
            world = parser_obj.parse(text)
            stem = _os.path.splitext(req.filename)[0]
            if not world.get("name") or world.get("name") in ("Unknown", "parsed_module", ""):
                world["name"] = stem
            path = save_world_book(world.get("name", stem), world)
            _feed_to_rag(world)
            _parse_jobs[upload_id] = {
                "status": "done", "progress": 100,
                "world_name": world.get("name"),
                "path": path,
                "scene_count": len(world.get("scenes", {})),
                "entity_count": len(world.get("entities", {})),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            try:
                if hasattr(parser_obj, '_last_error') and parser_obj._last_error:
                    error_msg = f"{error_msg} | LLM: {parser_obj._last_error}"
            except Exception:
                pass
            _parse_jobs[upload_id] = {"status": "error", "error": error_msg}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "upload_id": upload_id}


@app.post("/api/parse/text")
def parse_text(req: ParseTextRequest, fast_request: Request):
    """One-step: receive raw text + filename, save, trigger parsing, return immediately."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    ext = _os.path.splitext(req.filename)[1].lower()
    if ext not in (".txt", ".md", ".docx", ".pdf"):
        ext = ".txt"

    upload_id = uuid.uuid4().hex[:12]
    save_path = _os.path.join(UPLOADS_DIR, f"{upload_id}{ext}")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(req.text)

    auth_header = fast_request.headers.get("Authorization", "")
    api_key = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    from config import DEEPSEEK_API_KEY
    print(f"[AIKP] Key check: header='{api_key[:5] if api_key else 'empty'}', config='{DEEPSEEK_API_KEY[:5] if DEEPSEEK_API_KEY else 'empty'}'", flush=True)
    if not api_key:
        api_key = DEEPSEEK_API_KEY
    print(f"[AIKP] After config fallback: '{api_key[:5] if api_key else 'empty'}'", flush=True)
    if not api_key:
        print("[AIKP] Calling _read_st_secret...", flush=True)
        api_key = _read_st_secret()
        if api_key:
            print(f"[AIKP] Using key from SillyTavern secrets", flush=True)
    if not api_key:
        print("[AIKP] No API key from any source (header/config/secrets)", flush=True)
        raise HTTPException(status_code=400, detail="No API key configured (checked: header, config.py, secrets.json)")

    _parse_jobs[upload_id] = {"status": "running", "progress": 0, "step": "starting"}

    def _run_parse():
        try:
            from parser import ModuleParser, load_upload_text, save_world_book
            _parse_jobs[upload_id] = {"status": "running", "progress": 5, "step": "parsing"}
            parser_obj = ModuleParser(api_key=api_key)
            world = parser_obj.parse(req.text)
            # Use filename stem as fallback name if parser couldn't extract a title
            stem = _os.path.splitext(req.filename)[0]
            if not world.get("name") or world.get("name") in ("Unknown", "parsed_module", ""):
                world["name"] = stem
            path = save_world_book(world.get("name", stem), world)
            _parse_jobs[upload_id] = {
                "status": "done", "progress": 100,
                "world_name": world.get("name"),
                "path": path,
                "scene_count": len(world.get("scenes", {})),
                "entity_count": len(world.get("entities", {})),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            try:
                if hasattr(parser_obj, '_last_error') and parser_obj._last_error:
                    error_msg = f"{error_msg} | LLM: {parser_obj._last_error}"
            except Exception:
                pass
            _parse_jobs[upload_id] = {"status": "error", "error": error_msg}

    threading.Thread(target=_run_parse, daemon=True).start()
    return {"status": "started", "upload_id": upload_id}


@app.post("/api/parse/{upload_id}")
def parse_module(upload_id: str, fast_request: Request):
    from parser import ModuleParser, load_upload_text, save_world_book

    if upload_id in _parse_jobs and _parse_jobs[upload_id].get("status") == "running":
        return {"status": "already_running", "upload_id": upload_id}

    auth_header = fast_request.headers.get("Authorization", "")
    api_key = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    from config import DEEPSEEK_API_KEY
    if not api_key:
        api_key = DEEPSEEK_API_KEY
    if not api_key:
        api_key = _read_st_secret()
    if not api_key:
        raise HTTPException(status_code=400, detail="No API key configured")

    _parse_jobs[upload_id] = {"status": "running", "progress": 0, "step": "starting"}

    def _run_parse():
        try:
            _parse_jobs[upload_id] = {"status": "running", "progress": 5, "step": "loading text"}
            text = load_upload_text(upload_id)
            _parse_jobs[upload_id] = {"status": "running", "progress": 10, "step": "parsing"}
            parser_obj = ModuleParser(api_key=api_key)
            world = parser_obj.parse(text)
            path = save_world_book(world.get("name", "parsed_module"), world)
            _parse_jobs[upload_id] = {
                "status": "done", "progress": 100,
                "world_name": world.get("name"),
                "path": path,
                "scene_count": len(world.get("scenes", {})),
                "entity_count": len(world.get("entities", {})),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            try:
                if hasattr(parser_obj, '_last_error') and parser_obj._last_error:
                    error_msg = f"{error_msg} | LLM: {parser_obj._last_error}"
            except Exception:
                pass
            _parse_jobs[upload_id] = {"status": "error", "error": error_msg}

    threading.Thread(target=_run_parse, daemon=True).start()
    return {"status": "started", "upload_id": upload_id}


@app.get("/api/parse/{upload_id}/status")
def parse_status(upload_id: str):
    job = _parse_jobs.get(upload_id)
    if not job:
        raise HTTPException(status_code=404, detail="No parse job found for this upload_id")
    return job


@app.get("/parse", response_class=HTMLResponse)
def parse_page():
    """Page to trigger parsing and preview results."""
    return HTMLResponse("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIKP - Parse Module</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; max-width: 800px; margin: 0 auto; }
h1 { font-size: 1.4em; margin-bottom: 4px; } h1 span { color: #e94560; }
.card { background: #16213e; border-radius: 8px; padding: 16px; margin: 12px 0; }
label { display: block; margin-bottom: 4px; color: #aaa; font-size: .85em; }
input, button { width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #333; background: #0a0a1a; color: #e0e0e0; font-size: .95em; margin-bottom: 10px; }
button { background: #e94560; color: #fff; border: none; cursor: pointer; font-weight: bold; }
button:hover { background: #d63850; }
button:disabled { background: #555; cursor: not-allowed; }
#status { margin-top: 8px; padding: 10px; border-radius: 6px; display: none; }
#status.running { display: block; background: #0f3460; color: #4ecca3; }
#status.done { display: block; background: #0f3460; color: #4ecca3; }
#status.error { display: block; background: #3a0a0a; color: #e94560; }
#preview { background: #0a0a1a; border-radius: 6px; padding: 12px; font-size: .85em; max-height: 400px; overflow-y: auto; white-space: pre-wrap; display: none; margin-top: 8px; }
.bar { height: 4px; background: #333; border-radius: 2px; margin-top: 8px; }
.bar-fill { height: 100%; background: #e94560; border-radius: 2px; transition: width .3s; }
.poll-hint { color: #666; font-size: .8em; margin-top: 6px; }
</style>
</head>
<body>
<h1>AIKP <span>Parse Module</span></h1>
<div class="card">
  <label>Upload ID</label>
  <input id="uploadId" placeholder="Paste upload_id from upload page">
  <button id="parseBtn" onclick="startParse()">Start Parse</button>
  <div id="status"></div>
  <div class="bar"><div id="progressBar" class="bar-fill" style="width:0"></div></div>
  <div class="poll-hint" id="pollHint" style="display:none">Parsing in background. Polling every 2s...</div>
</div>
<div class="card" id="resultCard" style="display:none">
  <h3>Parse Result</h3>
  <pre id="preview"></pre>
</div>
<script>
let pollTimer = null;

async function startParse() {
  const id = document.getElementById('uploadId').value.trim();
  if (!id) return alert('Enter upload_id first');

  document.getElementById('parseBtn').disabled = true;
  const status = document.getElementById('status');
  status.className = 'running';
  status.textContent = 'Starting...';
  status.style.display = 'block';
  document.getElementById('pollHint').style.display = 'block';

  try {
    const r = await fetch('/api/parse/' + id, { method: 'POST' });
    const d = await r.json();
    status.textContent = d.status === 'already_running' ? 'Already running, polling...' : 'Started. Parsing...';
    startPoll(id);
  } catch(e) {
    status.className = 'error';
    status.textContent = 'Error: ' + e.message;
    document.getElementById('parseBtn').disabled = false;
  }
}

function startPoll(id) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const r = await fetch('/api/parse/' + id + '/status');
      const d = await r.json();
      const status = document.getElementById('status');
      const bar = document.getElementById('progressBar');
      bar.style.width = (d.progress || 0) + '%';

      if (d.status === 'done') {
        clearInterval(pollTimer);
        status.className = 'done';
        status.textContent = 'Done! ' + d.world_name + ' (' + d.scene_count + ' scenes, ' + d.entity_count + ' entities)';
        document.getElementById('parseBtn').disabled = false;
        document.getElementById('pollHint').style.display = 'none';
        document.getElementById('resultCard').style.display = 'block';
        document.getElementById('preview').style.display = 'block';
        document.getElementById('preview').textContent = JSON.stringify(d, null, 2);
      } else if (d.status === 'error') {
        clearInterval(pollTimer);
        status.className = 'error';
        status.textContent = 'Error: ' + d.error;
        document.getElementById('parseBtn').disabled = false;
      } else {
        status.className = 'running';
        status.textContent = 'Parsing... ' + (d.step || '') + ' (' + (d.progress || 0) + '%)';
      }
    } catch(e) {
      document.getElementById('status').textContent = 'Poll error: ' + e.message;
    }
  }, 2000);
}
</script>
</body>
</html>""")


if __name__ == "__main__":
    import uvicorn

    @app.on_event("startup")
    def _scan_data_bank():
        """Scan SillyTavern Data Bank files on startup, auto-parse uncached ones."""
        def _scan():
            st_files = _os.path.normpath(_os.path.join(
                BACKEND_DIR, "..", "Tavern", "SillyTavern",
                "data", "default-user", "files"
            ))
            if not _os.path.isdir(st_files):
                return
            api_key = _read_st_secret()
            from config import DEEPSEEK_API_KEY as dk
            if not api_key:
                api_key = dk
            if not api_key:
                print("[AIKP] Startup scan skipped — no API key", flush=True)
                return
            from parser import ModuleParser, save_world_book
            parser_obj = ModuleParser(api_key=api_key)
            for fname in sorted(_os.listdir(st_files)):
                if not fname.endswith(".txt"):
                    continue
                fpath = _os.path.join(st_files, fname)
                # Derive world book name from file (use filename stem)
                stem = _os.path.splitext(fname)[0]
                # Check if already cached as world book
                model_name = None
                for wf in _os.listdir(WORLD_BOOK_DIR):
                    if wf.endswith(".json"):
                        model_name = wf[:-5]
                        break
                if model_name:
                    print(f"[AIKP] Startup: models/ already has '{model_name}', skipping scan", flush=True)
                    return
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                    if len(text) < 50:
                        continue
                    print(f"[AIKP] Startup: auto-parsing {fname} ({len(text)} chars)...", flush=True)
                    world = parser_obj.parse(text)
                    world["name"] = world.get("name", stem)
                    save_world_book(world.get("name", stem), world)
                    print(f"[AIKP] Startup: world book saved to models/", flush=True)
                except Exception as e:
                    print(f"[AIKP] Startup scan error for {fname}: {e}", flush=True)

        threading.Thread(target=_scan, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8001)
