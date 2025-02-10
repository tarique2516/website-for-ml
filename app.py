import os
import pickle
import re
import docx
import pdfplumber
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer


# Extract text from the uploaded file (PDF or DOCX)
def extract_text_from_file(file):
    text = ""
    if file.filename.endswith(".pdf"):
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"[ERROR] PDF Processing Failed: {e}")
            return None
    elif file.filename.endswith(".docx"):
        try:
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"[ERROR] DOCX Processing Failed: {e}")
            return None
    else:
        print("[ERROR] Unsupported file format!")
        return None

    return text.strip() if text.strip() else None


# Extract skills from the resume text by searching for a "Skills:" section
def extract_skills_from_text(text):
    # This regex looks for a line starting with "Skills:" (case-insensitive)
    pattern = re.compile(r"skills\s*:\s*(.*)", re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        # Assuming the first occurrence is the skills list
        skills_line = matches[0]
        # Split the skills by comma, semicolon, or newline and remove extra whitespace
        skills = re.split(r",|;|\n", skills_line)
        skills = [skill.strip() for skill in skills if skill.strip()]
        return skills
    else:
        return []


# Load ML model and TF-IDF vectorizer from disk
def load_model_and_vectorizer():
    model_path = "model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, "rb") as model_file:
            rf = pickle.load(model_file)
        with open(vectorizer_path, "rb") as vectorizer_file:
            tfidf = pickle.load(vectorizer_file)
        print("[INFO] Model and Vectorizer loaded successfully.")
        return rf, tfidf
    else:
        print("[ERROR] Model or Vectorizer missing!")
        return None, None


# Process the uploaded resume: extract text, extract skills, vectorize the text, and predict the job/skill
def process_resume(file):
    rf, tfidf = load_model_and_vectorizer()

    if rf is None or tfidf is None:
        return "[ERROR] ML model is missing!", None, None

    text = extract_text_from_file(file)
    if not text:
        return "[ERROR] Invalid or unsupported file format!", None, None

    # Extract skills from the text
    extracted_skills = extract_skills_from_text(text)

    try:
        # Transform text using the TF-IDF vectorizer
        text_vectorized = tfidf.transform([text])
        print("Vectorized Input Shape:", text_vectorized.shape)

        # Predict the job/skill using the ML model
        predicted_job = rf.predict(text_vectorized)[0]
        print("Predicted Job:", predicted_job)

        return None, predicted_job, extracted_skills
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return f"[ERROR] Prediction failed: {e}", None, extracted_skills


app = Flask(__name__, template_folder="templates")


@app.route("/", methods=["GET", "POST"])
def index():
    predicted_job = None
    error_message = None
    extracted_skills = []

    if request.method == "POST":
        if "resume" not in request.files:
            error_message = "No file uploaded!"
        else:
            file = request.files["resume"]
            if file.filename == "":
                error_message = "No selected file!"
            else:
                error_message, predicted_job, extracted_skills = process_resume(
                    file)

    return render_template("index.html",
                           predicted_job=predicted_job or "",
                           error_message=error_message or "",
                           extracted_skills=extracted_skills)


if __name__ == "__main__":
    app.run(debug=True)
