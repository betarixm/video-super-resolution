import uuid

from flask import session


def auth() -> str:
    if "id" not in session:
        session["id"] = uuid.uuid4().hex
    return session["id"]
