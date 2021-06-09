import os.path

from flask import Flask, render_template, redirect, send_file
from werkzeug.utils import secure_filename

import environ as env
import session as session_m
import file as file_m


app = Flask(__name__)
app.secret_key = "SAFE_SECRET_KEY_HERE"
app.config["UPLOAD_FOLDER"] = env.UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = env.PROCESSED_FOLDER


@app.route("/", methods=["GET"])
def index():
    session_id = session_m.auth()
    return render_template("index.html", file_list=file_m.listing(session_id).items())


@app.route("/upload", methods=["POST"])
def upload():
    session_id = session_m.auth()
    file_m.save(session_id)
    return redirect("/")


@app.route("/download/<category>/<filename>", methods=["GET"])
def download(category, filename):
    session_id = session_m.auth()
    return send_file(os.path.join(".", secure_filename(category), session_id, secure_filename(filename)))


if __name__ == "__main__":
    app.run(debug=True)
