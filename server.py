import utils.session
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import re
import pdfplumber
import spacy
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher
from collections import defaultdict
import time

app = Flask(__name__)
CORS(app)

# === Enhanced Global Skill Dictionary 2025 ===
SKILLS = {
    # --- Information Technology & Software (Programming, Dev, CS/CSE) ---
    "python": 1.3, "java": 1.3, "c++": 1.3, "c#": 1.2, "golang": 1.2,
    "php": 1.1, "ruby": 1.1, "typescript": 1.2, "javascript": 1.2,
    "kotlin": 1.1, "swift": 1.1, "r": 1.1, "matlab": 1.1, "rust": 1.2,

    "frontend": 1.3, "backend": 1.3, "fullstack": 1.3,
    "react": 1.3, "vue": 1.2, "angular": 1.2, "next.js": 1.2, "nuxt.js": 1.1,
    "html": 1.1, "css": 1.1, "bootstrap": 1.1, "tailwind": 1.1,
    "node.js": 1.3, "express": 1.2, "django": 1.2, "flask": 1.2,
    "fastapi": 1.2, "spring boot": 1.2, "dotnet": 1.2,

    "sql": 1.2, "mysql": 1.2, "postgresql": 1.2, "oracle": 1.1,
    "mongodb": 1.2, "firebase": 1.1, "redis": 1.1, "cassandra": 1.1,

    "devops": 1.3, "ci/cd": 1.3, "docker": 1.3, "kubernetes": 1.3,
    "jenkins": 1.2, "terraform": 1.2, "ansible": 1.1,
    "aws": 1.3, "azure": 1.3, "gcp": 1.3, "linux": 1.2,

    "cybersecurity": 1.3, "ethical hacking": 1.3,
    "penetration testing": 1.3, "network security": 1.2,
    "cryptography": 1.2, "forensics": 1.1, "firewalls": 1.1,

    "ai": 1.3, "ml": 1.3, "deep learning": 1.3,
    "nlp": 1.3, "computer vision": 1.2,
    "data science": 1.3, "data analysis": 1.3,
    "chatgpt": 1.2, "llm": 1.2,
    "pandas": 1.1, "numpy": 1.1, "scikit-learn": 1.1,
    "tensorflow": 1.2, "pytorch": 1.2, "matplotlib": 1.1, "seaborn": 1.1,
    "big data": 1.2, "hadoop": 1.1, "spark": 1.1,

    "sqa": 1.3, "quality assurance": 1.3,
    "manual testing": 1.2, "automation testing": 1.2,
    "selenium": 1.2, "junit": 1.1, "pytest": 1.1, "testng": 1.1,
    "api testing": 1.2, "load testing": 1.1,

    "operating systems": 1.1, "computer networks": 1.2,
    "tcp/ip": 1.1, "cloud computing": 1.2, "distributed systems": 1.2,

    "software engineering": 1.3, "web development": 1.2,

    "project manager": 1.3, "scrum master": 1.2,
    "product manager": 1.3, "system analyst": 1.2,
    "business analyst": 1.2, "it consultant": 1.2,
    "software architect": 1.3, "tech lead": 1.2,

    # --- Healthcare & Medicine ---
    "doctor": 1.3, "nurse": 1.2, "pharmacist": 1.2,
    "physiotherapist": 1.2, "medical researcher": 1.3, "public health": 1.1,

    # --- Engineering & Technical ---
    "civil engineering": 1.2, "electrical engineering": 1.2,
    "mechanical engineering": 1.2, "biomedical engineering": 1.3,
    "chemical engineering": 1.2, "cad": 1.1,

    # --- Education & Training ---
    "teacher": 1.2, "lecturer": 1.2, "professor": 1.3,
    "academic researcher": 1.3, "corporate trainer": 1.2, "pedagogy": 1.1,

    # --- Business, Management & Administration ---
    "manager": 1.2, "hr": 1.1, "project management": 1.2,
    "operations management": 1.2, "business analysis": 1.2,
    "consultant": 1.2, "entrepreneurship": 1.3, "leadership": 1.2,

    # --- Finance, Banking & Accounting ---
    "accountant": 1.2, "auditor": 1.2, "investment banker": 1.3,
    "tax consultant": 1.2, "financial analyst": 1.3,
    "risk management": 1.2, "economics": 1.2, "bookkeeping": 1.1,

    # --- Law, Security & Government ---
    "lawyer": 1.3, "judge": 1.3, "public administration": 1.2,
    "diplomacy": 1.2, "governance": 1.1, "policy making": 1.2,

    # --- Creative, Arts & Media ---
    "journalism": 1.3, "author": 1.2, "marketing": 1.2,
    "brand strategy": 1.2, "copywriting": 1.1, "storytelling": 1.1,
    "graphic design": 1.2, "digital media": 1.2
}

# === Context-Aware Skill Mapping ===
SKILL_CONTEXT_MAP = {
    # Programming
    "c++": ["cpp"], "c#": ["csharp", ".net"],
    "python": ["py"], "java": ["jdk", "jre"],
    "javascript": ["js"], "typescript": ["ts"],
    "dotnet": [".net", "asp.net"],

    # Web Dev
    "frontend": ["frontend developer", "ui developer"],
    "backend": ["backend developer", "server-side"],
    "fullstack": ["full stack developer"],
    "react": ["reactjs", "react.js"],
    "vue": ["vuejs", "vue.js"],
    "angular": ["angularjs"],
    "node.js": ["nodejs", "node"],
    "express": ["expressjs"],
    "django": ["django framework"],
    "flask": ["flask framework"],
    "fastapi": ["python api"],
    "web development": ["web developer", "website development"],

    # Databases
    "mysql": ["maria db"],
    "postgresql": ["postgres"],
    "mongodb": ["mongo"],
    "oracle": ["oracle db"],
    "redis": ["redis cache"],

    # DevOps
    "ci/cd": ["continuous integration", "continuous deployment"],
    "docker": ["containerization"],
    "kubernetes": ["k8s"],
    "aws": ["amazon web services"],
    "azure": ["microsoft azure"],
    "gcp": ["google cloud"],

    # Cybersecurity
    "cybersecurity": ["infosec", "network security", "information security"],
    "ethical hacking": ["penetration tester", "red teaming"],
    "forensics": ["digital forensics"],

    # AI / ML
    "ai": ["artificial intelligence"],
    "ml": ["machine learning"],
    "nlp": ["natural language processing"],
    "deep learning": ["neural networks"],
    "chatgpt": ["openai", "gpt", "language model"],
    "data science": ["data scientist", "big data"],
    "data analysis": ["data analyst"],

    # QA / Testing
    "sqa": ["software testing", "qa engineer"],
    "automation testing": ["automated tests"],
    "manual testing": ["manual tester"],
    "selenium": ["selenium webdriver"],
    "api testing": ["postman"],

    # Networking
    "computer networks": ["networking"],
    "tcp/ip": ["network protocol"],

    # Management & Analyst
    "project manager": ["pmp", "agile manager"],
    "scrum master": ["agile coach"],
    "product manager": ["product owner"],
    "system analyst": ["systems analysis"],
    "business analyst": ["ba"],
    "software architect": ["solution architect"],
    "tech lead": ["technical lead", "team lead"],
    "project management": ["project coordination", "program management"],

    # Healthcare
    "doctor": ["physician", "medical doctor", "md"],
    "nurse": ["registered nurse", "rn", "clinical nurse"],
    "pharmacist": ["pharmacy"],
    "physiotherapist": ["physical therapist"],
    "medical researcher": ["clinical researcher", "biomedical research"],
    "public health": ["epidemiology", "community health"],

    # Engineering
    "civil engineering": ["civil engineer", "structural engineer"],
    "electrical engineering": ["electrical engineer", "electronics engineer"],
    "mechanical engineering": ["mechanical engineer"],
    "biomedical engineering": ["biomedical engineer", "bioengineering"],
    "chemical engineering": ["chemical engineer", "process engineer"],
    "cad": ["autocad", "solidworks"],
    "matlab": ["simulink"],

    # Education
    "teacher": ["school teacher", "educator"],
    "lecturer": ["academic lecturer"],
    "professor": ["faculty", "academician"],
    "academic researcher": ["research scholar"],
    "corporate trainer": ["learning consultant", "training specialist"],

    # Business & Management
    "manager": ["project manager", "operations manager", "hr manager"],
    "business analysis": ["business analyst"],
    "consultant": ["business consultant", "management consultant"],
    "entrepreneurship": ["startup founder"],
    "leadership": ["team lead", "executive leadership"],

    # Finance
    "accountant": ["chartered accountant", "ca"],
    "auditor": ["internal auditor", "external auditor"],
    "investment banker": ["ib", "equity banker"],
    "tax consultant": ["tax advisor"],
    "financial analyst": ["equity analyst", "finance analyst"],
    "risk management": ["risk analyst", "risk officer"],

    # Law & Government
    "lawyer": ["attorney", "advocate", "barrister"],
    "judge": ["magistrate"],
    "public administration": ["civil servant"],
    "diplomacy": ["foreign service", "ambassador"],
    "policy making": ["policy advisor"],

    # Creative & Media
    "journalism": ["journalist", "reporter"],
    "author": ["writer", "novelist"],
    "marketing": ["digital marketing", "seo specialist"],
    "brand strategy": ["brand strategist", "branding"],
    "copywriting": ["content writer"],
    "graphic design": ["designer", "ui/ux"],
    "digital media": ["social media specialist", "content creator"]
}

# ✅ Precompile regex for performance
PRECOMPILED_PATTERNS = {
    term: [re.compile(rf'\b{re.escape(v)}\b', re.I) for v in variations]
    for term, variations in SKILL_CONTEXT_MAP.items()
}

# Load SpaCy model once
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp.make_doc(skill) for skill in SKILLS.keys()]
matcher.add("SKILLS", patterns)

def extract_text_from_file(filepath):
    if filepath.endswith('.pdf'):
        with pdfplumber.open(filepath) as pdf:
            return " ".join([page.extract_text() or "" for page in pdf.pages])
    elif filepath.endswith(('.doc', '.docx')):
        doc = docx.Document(filepath)
        return " ".join(p.text for p in doc.paragraphs if p.text)
    raise ValueError("Unsupported file format")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s+.#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).lower().strip()
    return text


def extract_skills(text):
    text = clean_text(text)
    doc = nlp(text)
    matches = matcher(doc)
    found = set()

    for _, start, end in matches:
        skill = doc[start:end].text.strip().lower()
        if skill in SKILLS:
            found.add(skill)

    for term, regex_list in PRECOMPILED_PATTERNS.items():
        if any(r.search(text) for r in regex_list):
            found.add(term)

    return found

def calculate_similarity(job_desc, resume_text):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf = vectorizer.fit_transform([job_desc, resume_text])
    semantic_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

    job_skills = extract_skills(job_desc)
    resume_skills = extract_skills(resume_text)
    matched = job_skills & resume_skills
    missing = job_skills - resume_skills

    # ✅ Weighted scoring: 60% skills, 40% semantic
    skill_score = (len(matched) / max(1, len(job_skills))) * 100
    final_score = (skill_score * 0.6) + (semantic_score * 0.4)

    # ✅ Explainability: capture sentences mentioning missing skills
    job_lines = job_desc.split('.')
    why_missing = {}
    for skill in missing:
        related_lines = [line.strip() for line in job_lines if skill in line.lower()]
        if related_lines:
            why_missing[skill] = related_lines[:2]  # up to 2 examples

    return final_score, matched, missing, why_missing

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    file = request.files['resume']
    job_desc = request.form.get('job_description', '').strip()
    if not job_desc:
        return jsonify({"error": "Job description required"}), 400

    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', secure_filename(file.filename))
    file.save(filepath)

    try:
        resume_text = extract_text_from_file(filepath)
        score, matched, missing, why_missing = calculate_similarity(job_desc, resume_text)
        suggestions = "Focus on: " + ", ".join(sorted(missing)[:3]) if missing else "Excellent match!"

        return jsonify({
            "match_score": round(score, 1),
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
            "why_missing": why_missing,
            "suggestions": suggestions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(port=5000, debug=True)