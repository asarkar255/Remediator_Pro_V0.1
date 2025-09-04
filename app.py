# app_remediate_rag_chroma.py
import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# LangChain + OpenAI + Chroma
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# =========================
# Config (env)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # e.g., "gpt-5"
CHROMA_DIR   = os.getenv("CHROMA_DIR", "chroma_rules")  # persisted vector store
RAG_RULES_DIR= os.getenv("RAG_RULES_DIR", "rag_rules")  # put your rules *.txt/*.md here
RAG_TOP_K    = int(os.getenv("RAG_TOP_K", "6"))
RAG_REBUILD_ON_START = os.getenv("RAG_REBUILD_ON_START", "false").lower() == "true"

# Optional LangSmith tracing (if you use it)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="ABAP Remediator (LLM + Chroma RAG, PwC tagging)",
    version="2.0"
)

# =========================
# Models
# =========================
class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    code: str
    llm_prompt: List[str] = Field(default_factory=list)

class RebuildResponse(BaseModel):
    ok: bool
    docs_indexed: int
    persist_directory: str

# =========================
# RAG helpers (Chroma)
# =========================
def _load_rule_docs() -> List[Document]:
    rules_path = Path(RAG_RULES_DIR)
    if not rules_path.exists():
        raise RuntimeError(f"RAG rules directory '{RAG_RULES_DIR}' does not exist.")
    docs: List[Document] = []
    for f in rules_path.glob("*.*"):
        if f.suffix.lower() in {".txt", ".md"}:
            txt = f.read_text(encoding="utf-8")
            if txt.strip():
                docs.append(Document(page_content=txt, metadata={"source": str(f)}))
    return docs

def rebuild_rag_index() -> int:
    docs = _load_rule_docs()
    if not docs:
        raise RuntimeError(f"No rule files found in '{RAG_RULES_DIR}'. Place *.txt/*.md rules there.")
    embeddings = OpenAIEmbeddings()
    # Overwrite existing index
    Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    return len(docs)

def _get_retriever():
    embeddings = OpenAIEmbeddings()
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vs.as_retriever(search_kwargs={"k": RAG_TOP_K})

# =========================
# LLM prompt (strict JSON)
# =========================
SYSTEM_MSG = (
    "You are a precise ABAP remediation engine. "
    "Use the retrieved rules verbatim. Follow the bullets in 'llm_prompt' exactly. "
    "Return STRICT JSON only."
)

USER_TEMPLATE = """
<retrieved_rules>
{rules}
</retrieved_rules>

Remediate the ABAP code EXACTLY following the bullet points in 'llm_prompt'.
Rules you MUST follow:
- Replace legacy/wrong code with corrected ABAP per the rules and bullets.
- Output the FULL remediated code (not a diff).
- Every ADDED or MODIFIED line must include an inline ABAP comment at the end of that line:  " Added By Pwc{today_date}
  (Use a single double-quote ABAP comment delimiter.)
- Keep behavior the same unless the bullets say otherwise.
- Use ECC-safe syntax unless the bullets allow otherwise.
- Return ONLY strict JSON with keys:
{{
  "remediated_code": "<full updated ABAP code with PwC comments on added/modified lines>"
}}

Context:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}
- Today's date (PwC tag): {today_date}

Original ABAP code:
{code}

llm_prompt (bullets):
{llm_prompt_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

# =========================
# LLM runner
# =========================
def today_iso() -> str:
    return datetime.date.today().isoformat()

def _extract_json_str(s: str) -> str:
    """
    Best-effort extractor: some LLMs may wrap JSON in code fences.
    """
    t = s.strip()
    if t.startswith("```"):
        # remove leading fence
        t = t.split("```", 2)
        if len(t) == 3:
            t = t[1] if not t[1].lstrip().startswith("{") else t[1]
            # if a language hint exists (`json`), drop the first line
            t = "\n".join(line for line in t.splitlines() if not line.strip().lower().startswith("json")).strip()
        else:
            t = s
    return t.strip()

def remediate_with_rag(unit: Unit) -> str:
    if not unit.llm_prompt:
        raise HTTPException(status_code=400, detail="llm_prompt must be a non-empty list of instructions.")

    # Retrieve top-k rule chunks
    retriever = _get_retriever()
    query = "ABAP syntax rules and remediation patterns for SELECT SINGLE, field lists, AFLE amount declarations, MATNR, etc."
    chunks = retriever.get_relevant_documents(query)
    rules_text = "\n\n---\n\n".join(d.page_content for d in chunks) if chunks else ""

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
    payload = {
        "rules": rules_text or "No rules retrieved.",
        "pgm_name": unit.pgm_name,
        "inc_name": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name or "",
        "code": unit.code or "",
        "today_date": today_iso(),
        "llm_prompt_json": json.dumps(unit.llm_prompt, ensure_ascii=False, indent=2),
    }
    msgs = prompt.format_messages(**payload)
    resp = llm.invoke(msgs)

    content = resp.content or ""
    content = _extract_json_str(content)
    try:
        data = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model did not return valid JSON: {e}")

    rem = data.get("remediated_code", "")
    if not isinstance(rem, str) or not rem.strip():
        raise HTTPException(status_code=502, detail="Model returned empty or invalid 'remediated_code'.")
    return rem

# =========================
# Endpoints
# =========================
@app.post("/rebuild-rag", response_model=RebuildResponse)
def rebuild_rag():
    count = rebuild_rag_index()
    return RebuildResponse(ok=True, docs_indexed=count, persist_directory=str(Path(CHROMA_DIR).resolve()))

@app.post("/remediate")
def remediate(unit: Unit) -> Dict[str, Any]:
    """
    Input JSON:
      {
        "pgm_name": "...",
        "inc_name": "...",
        "type": "...",
        "name": "",
        "class_implementation": "",
        "code": "<ABAP code>",
        "llm_prompt": [ "...bullet...", "...bullet..." ]
      }

    Output JSON:
      original fields + "rem_code": "<full remediated ABAP>"
    """
    rem_code = remediate_with_rag(unit)
    obj = unit.model_dump()
    obj["rem_code"] = rem_code
    return obj

@app.get("/health")
def health():
    # check index presence
    has_index = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())
    # check rules dir presence
    rules_dir_exists = Path(RAG_RULES_DIR).exists()
    return {
        "ok": True,
        "model": OPENAI_MODEL,
        "chroma_dir": str(Path(CHROMA_DIR).resolve()),
        "rag_rules_dir": str(Path(RAG_RULES_DIR).resolve()),
        "has_index": has_index,
        "rules_dir_exists": rules_dir_exists,
        "rag_rebuild_on_start": RAG_REBUILD_ON_START,
        "rag_top_k": RAG_TOP_K,
    }

# =========================
# Optional: rebuild on start
# =========================
if RAG_REBUILD_ON_START:
    try:
        cnt = rebuild_rag_index()
        print(f"[RAG] Index rebuilt on start with {cnt} docs in {CHROMA_DIR}")
    except Exception as ex:
        print(f"[RAG] Rebuild on start failed: {ex}. You can POST /rebuild-rag later.")
