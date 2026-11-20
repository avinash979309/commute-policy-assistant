"""
The HTTP API. Routes are thin: check who is calling, then hand over to
assistant.py or ingest.py. Run it with:

    uvicorn app.main:app --reload

Swagger UI (good for the demo): http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import assistant, auth, config, ingest, store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # Load the sample policies on first start, so `docker compose up` is enough.
    ingest.seed_default_policies()
    store.warm_up()
    if not (config.GROQ_API_KEYS or config.GEMINI_API_KEYS):
        log.warning("No LLM API keys set: answers will quote the best policy section (offline mode).")
    yield


app = FastAPI(title="Commute Policy Assistant", lifespan=lifespan)


# ---------------------------------------------------------------- request bodies

class TokenRequest(BaseModel):
    employee_id: str


class AskRequest(BaseModel):
    # Only the question. There is deliberately NO company field: the company
    # comes from the token. Extra fields a client sends are simply ignored.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                               max_length=config.MAX_QUESTION_CHARS)]


# ---------------------------------------------------------------- routes

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/token")
def login(body: TokenRequest):
    """Mock login: employee ID in, signed JWT out (company comes from our directory)."""
    try:
        token = auth.issue_token(body.employee_id)
    except auth.AuthError as error:
        raise HTTPException(status_code=404, detail=str(error))
    return {"access_token": token, "token_type": "bearer"}


@app.post("/ask")
def ask(body: AskRequest, user: dict = Depends(auth.current_user)):
    result = assistant.answer_question(user["company"], body.question)
    status_code = 503 if result["status"] == "error" else 200
    return JSONResponse(result, status_code=status_code)


@app.get("/policy")
def active_policy(user: dict = Depends(auth.current_user)):
    """Which policy version is live for my company?"""
    active = store.get_active(user["company"])
    if active is None:
        raise HTTPException(status_code=404, detail="No policy is loaded for your company yet.")
    return {"company": user["company"], **active}


@app.post(
    "/admin/policies",
    # Tell Swagger UI to show a plain text box for the markdown file.
    openapi_extra={"requestBody": {"content": {"text/markdown": {"schema": {"type": "string"}}},
                                   "required": True}},
)
async def upload_policy(request: Request, user: dict = Depends(auth.current_user)):
    """Upload a policy file (markdown) for YOUR company. Admins only."""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only a transport admin can upload policies.")

    text = (await request.body()).decode("utf-8", errors="replace")
    try:
        policy = ingest.parse_policy(text)
        if policy["company"] != user["company"]:
            raise HTTPException(status_code=403, detail="You can only upload your own company's policy.")
        return ingest.ingest_policy(text)
    except ingest.IngestError as error:
        raise HTTPException(status_code=400, detail=str(error))


# ---------------------------------------------------------------- error handlers
# Every error the user sees is the same small JSON with a plain message.
# Never a stack trace, never a raw provider or database error.

def _error(status_code, message):
    return JSONResponse({"status": "error", "answer": message, "citations": []}, status_code=status_code)


# Starlette's class also covers FastAPI's HTTPException and unknown-URL 404s.
@app.exception_handler(StarletteHTTPException)
async def http_error(request, error):
    return _error(error.status_code, error.detail)


@app.exception_handler(RequestValidationError)
async def bad_request(request, error):
    if request.url.path == "/ask":
        return _error(422, f"Please type your question (1 to {config.MAX_QUESTION_CHARS} characters), "
                           'sent as JSON like {"question": "..."}.')
    return _error(422, "The request is missing a field or has a wrong value. Please check it and try again.")


@app.exception_handler(Exception)
async def unexpected_error(request, error):
    log.exception("Unexpected error")  # full details go to the server log only
    return _error(500, "Something went wrong on our side. Please try again in a minute, "
                       "or contact your transport desk.")
