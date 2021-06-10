import uuid

from flask import session, copy_current_request_context


def auth() -> str:
    if "id" not in session:
        session["id"] = uuid.uuid4().hex
    return session["id"]
