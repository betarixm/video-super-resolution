import os
import glob

from flask import request, session
from werkzeug.utils import secure_filename

import environ as env
from app import app


def save(session_id: str) -> bool:
    path_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)

    if not os.path.exists(path_dir):
        os.mkdir(path_dir)

    f = request.files["file"]
    filename = secure_filename(f.filename)

    try:
        f.save(os.path.join(path_dir, filename))

    except Exception as e:
        return False

    return True


def listing(session_id: str) -> dict:
    uploaded = glob.glob(f"{app.config['UPLOAD_FOLDER']}/{session_id}/*")
    processed = glob.glob(f"{app.config['PROCESSED_FOLDER']}/{session_id}/*")

    return {
        os.path.basename(f): ["upload", "processed" if f in processed else None] for f in uploaded
    }
