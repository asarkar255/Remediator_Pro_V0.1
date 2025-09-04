# app_remediate_rag_chroma.py
import os, json, datetime, re
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- OpenAI / LangChain ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser

# ========= FastAPI =========
app = FastAPI(title="ABAP Remediator + RAG (Chroma, ECC-safe)", version="1.0")

# ========= Models =========
class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    code: str
    llm_prompt: List[str] = Field(default_factory=list)

# ========= RAG store (Chroma persistent) =========
RAG_KB_DIR = os.getenv("RAG_KB_DIR", "knowledge")      # put .md, .txt, .pdf here
RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "rag_index") # persistent directory
RAG_COLLECTION = os.getenv("RAG_COLLECTION", "abap_syntax_rules")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore: Optional[Chroma] = None

def _load_docs_from_dir(folder: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(folder):
        return docs
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        try:
            if fname.lower().endswith((".md", ".txt")):
                docs.extend(TextLoader(path, encoding="utf-8").load())
            elif fname.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
        except Exception as e:
            print(f"[RAG] Skip {fname}: {e}")
    return docs

def _new_chroma() -> Chroma:
    os.makedirs(RAG_INDEX_DIR, exist_ok=True)
    return Chroma(
        collection_name=RAG_COLLECTION,
        embedding_function=embeddings,
        persist_directory=RAG_INDEX_DIR,
    )

def rebuild_rag_index() -> str:
    """Recreates the Chroma collection from scratch using files in RAG_KB_DIR."""
    global vectorstore
    docs = _load_docs_from_dir(RAG_KB_DIR)
    if not docs:
        raise RuntimeError(f"No docs found in {RAG_KB_DIR}. Add .md/.txt/.pdf files with ABAP rules.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Re-init a fresh collection
    vectorstore = _new_chroma()
    # Best-effort wipe existing collection by using unique ids (not strictly required)
    # Chroma's add_texts will append; for a clean rebuild, we can recreate the directory.
    # Here we remove the directory for a true rebuild.
    try:
        # Danger: fully delete existing index dir
        import shutil
        shutil.rmtree(RAG_INDEX_DIR, ignore_errors=True)
    except Exception as e:
        print(f"[RAG] Could not clear index dir, continuing: {e}")

    vectorstore = _new_chroma()
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    vectorstore.persist()
    return f"Indexed {len(chunks)} chunks from {len(docs)} files."

def load_rag_index() -> None:
    """Loads the existing Chroma collection (if present); creates empty if missing."""
    global vectorstore
    try:
        vectorstore = _new_chroma()
        # If empty, .persist() keeps it consistent
        vectorstore.persist()
    except Exception as e:
        print(f"[RAG] Failed to load/create Chroma index: {e}")
        vectorstore = None

load_rag_index()

def retrieve_rules(query: str, k: int = RAG_TOP_K) -> List[str]:
    if not vectorstore:
        return []
    try:
        hits = vectorstore.similarity_search(query, k=k)
        return [h.page_content.strip() for h in hits if h and h.page_content]
    except Exception as e:
        print(f"[RAG] retrieval failed: {e}")
        return []

# ========= LLM prompt with RAG =========
SYSTEM_MSG = (
    "You are a precise ABAP remediation engine for ECC syntax. "
    "Use the provided ABAP rules as authoritative. Output STRICT JSON only."
)

USER_TEMPLATE = """
You will remediate ABAP code EXACTLY following 'llm_prompt'.
Hard rules:
- Replace legacy code per the prompts; keep behavior unchanged unless told otherwise.
- Output the FULL remediated code (not a diff).
- Every ADDED or MODIFIED line must include an inline ABAP comment at end of the line:
  \" Added By Pwc{today_date}
- Use strictly ECC-safe ABAP syntax (no pseudo comments, no @DATA inline if not ECC-safe).
- Follow the ABAP rules below as authoritative.

Return ONLY strict JSON with keys:
{{
  "remediated_code": "<full updated ABAP code with PwC comments on added/modified lines>"
}}

Context:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}
- Today: {today_date}

Authoritative ABAP rules (retrieved):
-----
{abap_rules}
-----

Original ABAP code:
-----
{code}
-----

llm_prompt (exact remediation instructions):
-----
{llm_prompt_json}
-----
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()

def today_iso() -> str:
    return datetime.date.today().isoformat()

# --- Simple sanity checks for ABAP (keep minimal & non-blocking) ---
ABAP_BAD_PATTERNS = [
    r"^\s*DATA\s*\(\w+\)\s*=",    # inline data declarations (not ECC-safe)
    r"^\s*FIELD-SYMBOLS\s+<[^>]+>$",  # lonely field-symbols w/o period
]
def abap_sanity_hints(code: str) -> List[str]:
    hints = []
    lines = code.splitlines()
    for i, ln in enumerate(lines, start=1):
        for pat in ABAP_BAD_PATTERNS:
            if re.search(pat, ln):
                hints.append(f"Line {i}: Avoid modern inline / unsupported ECC syntax near: {ln.strip()[:80]}")
        if ln.strip() and not ln.strip().endswith(".") and not ln.strip().startswith("*") and not ln.strip().startswith('"'):
            # Many ABAP statements end with '.', but don't over-enforce (FUNCTION/FORM blocks etc.)
            if re.match(r"^(DATA|PARAMETERS|SELECT|LOOP|READ|MOVE|CALL|IF|ELSEIF|ELSE|ENDIF|ENDLOOP|PERFORM|WRITE|FIELD-SYMBOLS|CONSTANTS|TABLES|TYPES)\b", ln.strip(), re.IGNORECASE):
                hints.append(f"Line {i}: ABAP statement may require a period at end.")
    return hints

def build_query_for_rag(u: Unit) -> str:
    # Use prompt + short slice of code as retrieval query
    lp = "\n".join(u.llm_prompt) if u.llm_prompt else ""
    head = (u.code or "")[:2000]
    return f"ABAP remediation rules for ECC; ensure correct SELECT, LOOP, MOVE, DATA declarations. Prompts:\n{lp}\nCode:\n{head}"

def run_remediation(u: Unit) -> str:
    if not u.llm_prompt:
        raise HTTPException(status_code=400, detail="llm_prompt must be a non-empty list of instructions.")
    rules_snippets = retrieve_rules(build_query_for_rag(u), k=RAG_TOP_K)
    abap_rules = "\n---\n".join(rules_snippets) if rules_snippets else "No rules retrieved. Use conservative ECC-safe ABAP."

    payload = {
        "pgm_name": u.pgm_name,
        "inc_name": u.inc_name,
        "unit_type": u.type,
        "unit_name": u.name or "",
        "code": u.code or "",
        "today_date": today_iso(),
        "llm_prompt_json": json.dumps(u.llm_prompt, ensure_ascii=False, indent=2),
        "abap_rules": abap_rules
    }

    # Try a couple of times if JSON parsing fails
    attempts, last_err = 2, None
    for _ in range(attempts):
        try:
            out = (prompt | llm | parser).invoke(payload)
            rem = out.get("remediated_code", "")
            if not isinstance(rem, str) or not rem.strip():
                raise ValueError("Model returned empty or invalid 'remediated_code'.")
            return rem
        except Exception as e:
            last_err = e
    raise HTTPException(status_code=502, detail=f"LLM remediation failed: {last_err}")

# ========= API =========
@app.post("/rebuild-index")
def rebuild_index_endpoint():
    msg = rebuild_rag_index()
    return {"ok": True, "message": msg, "persist_directory": RAG_INDEX_DIR, "collection": RAG_COLLECTION}

@app.post("/remediate")
def remediate(unit: Unit) -> Dict[str, Any]:
    rem_code = run_remediation(unit)
    hints = abap_sanity_hints(rem_code)
    obj = unit.model_dump()
    obj["rem_code"] = rem_code
    obj["sanity_hints"] = hints  # informational only
    return obj

@app.post("/preview-rules")
def preview_rules(unit: Unit) -> Dict[str, Any]:
    snippets = retrieve_rules(build_query_for_rag(unit), k=RAG_TOP_K)
    return {"ok": True, "top_k": RAG_TOP_K, "snippets": snippets}

@app.get("/health")
def health():
    # Light probe: confirm Chroma can be opened/persisted
    ok = vectorstore is not None
    return {
        "ok": ok,
        "model": OPENAI_MODEL,
        "embed_model": EMBED_MODEL,
        "persist_directory": RAG_INDEX_DIR,
        "collection": RAG_COLLECTION
    }
