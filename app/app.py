import glob
import os.path

from flask import Flask, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename

import app.environ as env
import app.session as session_m
import app.file as file_m
import app.video as video


app = Flask(__name__)
app.secret_key = "SAFE_SECRET_KEY_HERE"
app.config["UPLOAD_FOLDER"] = env.UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = env.PROCESSED_FOLDER
app.config["PROCESSING_FOLDER"] = env.PROCESSING_FOLDER


@app.route("/", methods=["GET"])
def index():
    session_id = session_m.auth()
    return render_template("index.html", file_list=file_m.listing(session_id).items())


@app.route("/upload", methods=["POST"])
def upload():
    session_id = session_m.auth()
    file_m.save(session_id)
    return redirect("/")


@app.route("/download/<category>/<filename>", methods=["GET", "POST"])
def download(category, filename):
    session_id = session_m.auth()
    return send_from_directory(
        os.path.join(".", secure_filename(category), session_id),
        path=secure_filename(filename),
        as_attachment=True,
        mimetype="video/mp4",
        filename=secure_filename(filename)
    )


@app.route("/process/<filename>", methods=["POST"])
def process(filename: str):
    session_id = session_m.auth()
    video.process(session_id, filename)
    return redirect("/")


@app.route("/status/", methods=["GET", "POST"])
def status():
    session_id = session_m.auth()
    return file_m.listing(session_id)


@app.route("/status/<filename>", methods=["GET", "POST"])
def status_per_file(filename: str):
    session_id = session_m.auth()

    if filename in video.running:
        return "processing"
    elif filename in [os.path.basename(f) for f in glob.glob(f"{app.config['PROCESSED_FOLDER']}/{session_id}/*")]:
        return "processed"
    elif filename in [os.path.basename(f) for f in glob.glob(f"{app.config['UPLOAD_FOLDER']}/{session_id}/*")]:
        return "uploaded"
    else:
        return "none"


if __name__ == "__main__":
    for d in [env.UPLOAD_FOLDER, env.PROCESSING_FOLDER, env.PROCESSED_FOLDER]:
        if not os.path.exists(d):
            os.mkdir(d)

    app.run(debug=True)
