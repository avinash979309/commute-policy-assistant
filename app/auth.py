"""
Identity: who is asking, and which company they belong to.

The brief says company identity must come from the JWT only. So:
  * `issue_token()` is our mock "login": it looks the employee up in
    data/employees.json and puts THEIR company into a signed token.
    The caller can't choose the company.
  * `current_user()` is used by every protected route. It verifies the
    signature and expiry, and returns the claims. Routes read the company
    from here and nowhere else.
"""

import json
import time

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import config


def _read_json(file_name):
    return json.loads((config.DATA_DIR / file_name).read_text())


# Small mock directories, loaded once. Keyed by id for quick lookup.
EMPLOYEES = {e["employee_id"]: e for e in _read_json("employees.json")}
COMPANIES = {c["id"]: c for c in _read_json("companies.json")}


class AuthError(Exception):
    """Token missing, broken, expired or not trusted. Message is user-safe."""


def issue_token(employee_id):
    """Mock JWT issuer. In a real system this would be the company SSO."""
    employee = EMPLOYEES.get(employee_id)
    if employee is None:
        raise AuthError("We don't know this employee ID. Please check it and try again.")

    now = int(time.time())
    claims = {
        "sub": employee_id,
        "company": employee["company"],  # taken from OUR directory, not from the request
        "role": employee["role"],        # "employee" or "admin"
        "iat": now,
        "exp": now + config.JWT_EXPIRY_MINUTES * 60,
    }
    return jwt.encode(claims, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def verify_token(token):
    """Return the claims if the token is genuine and fresh, else raise AuthError."""
    try:
        claims = jwt.decode(
            token,
            config.JWT_SECRET,
            # Pinning the algorithm blocks the classic "alg: none" trick.
            algorithms=[config.JWT_ALGORITHM],
            options={"require": ["sub", "company", "role", "exp"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Your session has expired. Please sign in again.")
    except jwt.InvalidTokenError:
        raise AuthError("Your sign-in is not valid. Please sign in again.")

    if claims["company"] not in COMPANIES:
        raise AuthError("Your company is not set up for this assistant. Please contact the transport desk.")
    return claims


# auto_error=False so a missing header gets OUR plain message, not FastAPI's.
# Using HTTPBearer also gives Swagger UI its "Authorize" button for the demo.
_bearer = HTTPBearer(auto_error=False)


def current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    """FastAPI dependency: the verified token claims of the caller."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Please sign in first (missing Bearer token).")
    try:
        return verify_token(credentials.credentials)
    except AuthError as error:
        raise HTTPException(status_code=401, detail=str(error))
