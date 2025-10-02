from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import re
import spacy
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher
from collections import defaultdict
import numpy as np

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

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp.make_doc(skill) for skill in SKILLS.keys()]
matcher.add("SKILLS", patterns)

def extract_text_from_file(filepath):
    """Robust text extraction with error handling"""
    try:
        if filepath.endswith('.pdf'):
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
                return text
        elif filepath.endswith(('.doc', '.docx')):
            doc = docx.Document(filepath)
            return " ".join([para.text for para in doc.paragraphs if para.text])
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise RuntimeError(f"Text extraction failed: {str(e)}")

def clean_text(text):
    """Advanced text normalization"""
    # Preserve key tech terms (C++, .NET, etc.)
    text = re.sub(r'([a-zA-Z]+)\+', r'\1plus', text)  # Handle C++
    text = re.sub(r'\.([a-zA-Z]{2,})', r' dot \1', text)  # Handle .NET, .js
    
    # Remove special chars but preserve context
    text = re.sub(r'[^a-zA-Z0-9\s#+]', ' ', text)
    text = re.sub(r'\s+', ' ', text).lower().strip()
    
    # Restore preserved terms
    text = re.sub(r'(\w+)plus', r'\1+', text)
    text = re.sub(r' dot (\w+)', r'.\1', text)
    
    return text

def extract_skills(text):
    """Context-aware skill extraction with mapping"""
    text = clean_text(text)
    doc = nlp(text)
    matches = matcher(doc)
    found_skills = set()
    
    # Direct matches
    for _, start, end in matches:
        skill = doc[start:end].text
        found_skills.add(skill)
    
    # Context-based mapping
    for term, variations in SKILL_CONTEXT_MAP.items():
        for variant in variations:
            if variant in text:
                found_skills.add(term)
    
    # Handle compound terms (like "React.js")
    for token in doc:
        # Handle React.js -> react
        if token.text.endswith('.js') and token.text[:-3] in SKILLS:
            found_skills.add(token.text[:-3])
    
    return found_skills

def calculate_similarity(job_desc, resume_text):
    """Enhanced similarity with context weighting"""
    # TF-IDF similarity with focus on skills
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),  # Capture phrases like "machine learning"
        min_df=1,
        max_features=3000,
        sublinear_tf=True  # Boost rare terms
    )
    
    # Combine texts for better vectorization
    all_text = [job_desc, resume_text]
    tfidf = vectorizer.fit_transform(all_text)
    
    # Calculate base similarity
    base_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    
    # Skill-based weighting
    job_skills = extract_skills(job_desc)
    resume_skills = extract_skills(resume_text)
    matched_skills = job_skills & resume_skills
    missing_skills = job_skills - resume_skills
    
    # Calculate skill impact (max 40% boost)
    skill_impact = min(40, len(matched_skills) / max(1, len(job_skills)) * 40)
    
    # Final score: 10% text similarity, 90% skill match
    final_score = min(100, base_score * 0.1 + (len(matched_skills) / max(1, len(job_skills))) * 90)
    
    return final_score, matched_skills, missing_skills

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['resume']
    job_desc = request.form.get('job_description', '')
    
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400
        
    if not job_desc.strip():
        return jsonify({"error": "Job description required"}), 400

    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_dir, filename)
    
    try:
        file.save(filepath)
        resume_text = extract_text_from_file(filepath)
        
        # Calculate scores
        match_score, matched_skills, missing_skills = calculate_similarity(job_desc, resume_text)
        
        # Generate suggestions
        top_missing = sorted(missing_skills, key=lambda s: SKILLS.get(s, 1), reverse=True)[:3]
        suggestions = "Focus on: " + ", ".join(top_missing) if top_missing else "Strong match!"
        
        return jsonify({
            "match_score": round(match_score, 1),
            "matched_skills": sorted(matched_skills),
            "missing_skills": sorted(missing_skills),
            "suggestions": suggestions
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": f"File processing error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(port=5000, debug=True)