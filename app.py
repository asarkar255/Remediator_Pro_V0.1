# app_remediate_llm.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json, datetime

# ==== LLM is mandatory (no fallback) ====
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required (no fallback).")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="ABAP Remediator (LLM-driven, exact prompts, PwC tagging)", version="1.0")

# ========= Input model =========
class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    code: str
    llm_prompt: List[str] = Field(default_factory=list)

# ========= Prompt (strict JSON, exact rules) =========
SYSTEM_MSG = (
    "You are a precise ABAP remediation engine. "
    "You must follow the user-provided remediation prompts exactly and output strict JSON only."
)

USER_TEMPLATE = """
You will remediate ABAP code EXACTLY following the bullet points provided in 'llm_prompt'. 
Rules you MUST follow:
- Replace old/legacy code with the corrected version per the prompts.
- Output the FULL remediated code (not a diff).
- Every ADDED or MODIFIED line must include an inline ABAP comment at the end of that line:
  " Added By Pwc{today_date}
  The comment delimiter is the ABAP double quote character (").
- Keep behavior the same unless the prompts explicitly say otherwise.
- Do not add pseudo-comments, suppressions, or non-requested changes.
- Use strictly ECC-safe syntax unless a prompt allows otherwise.
- If a prompt requests a JSON output format for a later pipeline, keep that requirement in comments only if necessary.

Return ONLY strict JSON with keys:
{{
  "remediated_code": "<full updated ABAP code with PwC comments on added/modified lines>"
}}

Context metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

Today's date for PwC tag: {today_date}

Original ABAP code:
{code}

llm_prompt (bullet list of exact remediation instructions):
{llm_prompt_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def today_iso() -> str:
    # Use server's local date; format YYYY-MM-DD
    return datetime.date.today().isoformat()

def run_remediation(u: Unit) -> str:
    if not u.llm_prompt:
        raise HTTPException(status_code=400, detail="llm_prompt must be a non-empty list of instructions.")

    payload = {
        "pgm_name": u.pgm_name,
        "inc_name": u.inc_name,
        "unit_type": u.type,
        "unit_name": u.name or "",
        "code": u.code or "",
        "today_date": today_iso(),
        "llm_prompt_json": json.dumps(u.llm_prompt, ensure_ascii=False, indent=2)
    }
    try:
        out = chain.invoke(payload)
        rem = out.get("remediated_code", "")
        if not isinstance(rem, str) or not rem.strip():
            raise ValueError("Model returned empty or invalid 'remediated_code'.")
        return rem
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM remediation failed: {e}")

# ========= API =========
@app.post("/remediate")
async def remediate(unit: Unit) -> Dict[str, Any]:
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

    Output JSON: same fields + "rem_code" (remediated full code).
    The original fields must remain unchanged.
    """
    rem_code = run_remediation(unit)
    obj = unit.model_dump()
    obj["rem_code"] = rem_code
    return obj

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
