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

# Enhanced 2025 skills with context-aware mapping
SKILLS = {
    # Core Tech
    "python": 1.3, "react": 1.3, "node.js": 1.2, "typescript": 1.2, 
    "javascript": 1.1, "vue": 1.1, "next.js": 1.1,
    
    # AI/ML
    "llm": 1.4, "generative ai": 1.4, "langchain": 1.3, "chatgpt": 1.3,
    "ai": 1.2, "machine learning": 1.2, "nlp": 1.2,
    
    # Cloud/DevOps
    "aws": 1.3, "azure": 1.2, "gcp": 1.2, "docker": 1.2, 
    "ci/cd": 1.3, "jenkins": 1.1, "terraform": 1.1,
    
    # Web3/Blockchain
    "web3": 1.3, "blockchain": 1.2, "solidity": 1.2, "nft": 1.1,
    
    # Data
    "sql": 1.1, "mongodb": 1.1
}

# Context-aware skill mapping
SKILL_CONTEXT_MAP = {
    "react": ["reactjs", "react.js", "reactjs"],
    "vue": ["vuejs", "vue.js"],
    "node.js": ["nodejs", "node"],
    "typescript": ["ts"],
    "llm": ["large language model", "language model", "gpt", "chatgpt", "openai"],
    "ci/cd": ["continuous integration", "continuous deployment", "devops"],
    "aws": ["amazon web services"],
    "docker": ["containerization"],
    "web3": ["crypto", "cryptocurrency", "decentralized"],
    "solidity": ["smart contracts"],
    "fastapi": ["python api"]
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