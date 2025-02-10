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


def extract_skills_from_text(resume_text):
    # Define potential headings for the Skills section (case-insensitive)
    skill_headings = {
        'skills', 'technical skills', 'core competencies', 'key skills',
        'skills summary', 'technical proficiencies', 'areas of expertise',
        'competencies'
    }

    # Common section headings that typically follow Skills (case-insensitive)
    common_sections = {
        'work experience', 'experience', 'employment history', 'education',
        'academic background', 'projects', 'project work',
        'certifications & course', 'awards', 'hobbies', 'interests',
        'publications', 'volunteer experience', 'languages', 'references',
        'honors and awards', 'certifications', 'open source contributions',
        'objective'
    }

    lines = resume_text.split('\n')
    skills_section = []
    found_skills = False

    for line in lines:
        stripped_line = line.strip()
        lower_line = stripped_line.lower()

        # Detect Skills section start
        if not found_skills:
            if lower_line in skill_headings:
                found_skills = True
            continue

        # Stop when encountering a new section
        if lower_line in common_sections:
            break

        # Skip empty lines at the start of the Skills section
        if not skills_section and not stripped_line:
            continue

        skills_section.append(stripped_line)

    # Clean up trailing empty lines
    while skills_section and not skills_section[-1]:
        skills_section.pop()

    return skills_section


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
        return "[ERROR] ML model is missing!", None, None, None

    text = extract_text_from_file(file)
    if not text:
        return "[ERROR] Invalid or unsupported file format!", None, None, None

    # Extract skills from the text
    extracted_skills = extract_skills_from_text(text)

    try:
        # Transform text using the TF-IDF vectorizer
        text_vectorized = tfidf.transform([text])
        print("Vectorized Input Shape:", text_vectorized.shape)

        # Predict the job/skill using the ML model
        predicted_job = rf.predict(text_vectorized)[0]
        print("Predicted Job:", predicted_job)

        return None, predicted_job, extracted_skills, text
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return f"[ERROR] Prediction failed: {e}", None, extracted_skills, text


app = Flask(__name__, template_folder="templates")


@app.route("/", methods=["GET", "POST"])
def index():
    predicted_job = None
    error_message = None
    extracted_skills = []
    extracted_text = ""

    if request.method == "POST":
        if "resume" not in request.files:
            error_message = "No file uploaded!"
        else:
            file = request.files["resume"]
            if file.filename == "":
                error_message = "No selected file!"
            else:
                error_message, predicted_job, extracted_skills, extracted_text = process_resume(
                    file)

    return render_template("index.html",
                           predicted_job=predicted_job or "",
                           error_message=error_message or "",
                           extracted_skills=extracted_skills,
                           extracted_text=extracted_text)


if __name__ == "__main__":
    app.run(debug=True)
