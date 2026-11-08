import uuid
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from detector import detect_vehicles
from timing import calculate_timing

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me"  # needed for flash messages
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")

    if file is None or file.filename == "":
        flash("Please choose an image to upload.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Use JPG, PNG, BMP or WEBP.")
        return redirect(url_for("index"))

    # Random name so uploads never overwrite each other
    filename = f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower()}"
    file.save(UPLOAD_DIR / filename)

    return redirect(url_for("result", filename=filename))


@app.route("/result/<filename>")
def result(filename):
    if not (UPLOAD_DIR / filename).is_file():
        flash("That image was not found. Please upload again.")
        return redirect(url_for("index"))

    try:
        detection = detect_vehicles(UPLOAD_DIR / filename, RESULTS_DIR / filename)
    except ValueError:
        flash("Could not read that image. Please try a different file.")
        return redirect(url_for("index"))

    timing = calculate_timing(detection["counts"])

    return render_template("result.html", filename=filename, detection=detection, timing=timing)


@app.errorhandler(413)
def file_too_large(error):
    flash("File is too large. Maximum size is 10 MB.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
