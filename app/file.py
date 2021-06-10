import os
import glob

from flask import request, session
from werkzeug.utils import secure_filename

import video
import environ as env


def save(session_id: str) -> bool:
    path_dir = os.path.join(env.UPLOAD_FOLDER, session_id)

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
    uploaded = [os.path.basename(f) for f in glob.glob(f"{env.UPLOAD_FOLDER}/{session_id}/*")]
    processed = [os.path.basename(f) for f in glob.glob(f"{env.PROCESSED_FOLDER}/{session_id}/*")]

    return {
        f: [
            "upload",
            "processing" if f in video.running else None,
            "processed" if f in processed else None
        ] for f in uploaded
    }
