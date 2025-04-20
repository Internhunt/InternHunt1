import streamlit as st
import json
import requests
import pdfplumber
import nltk
import spacy
import re
import ollama
import matplotlib.pyplot as plt
from serpapi import GoogleSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    st.write("Debug: NLTK resources initialized")
except Exception as e:
    st.error(f"NLTK initialization failed: {e}")

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    st.write("Debug: spaCy model loaded")
except Exception as e:
    st.error(f"Failed to load spaCy model: {e}. Install with 'python -m spacy download en_core_web_sm'")
    nlp = None

# Check Ollama server
try:
    st.write("Debug: Checking Ollama server...")
    response = requests.get("http://localhost:11434", timeout=5)
    use_model = response.status_code == 200
    st.write("Debug: Ollama server is running" if use_model else "Debug: Ollama server not detected")
except Exception as e:
    use_model = False
    st.error(f"Ollama server check failed: {e}. Start with 'ollama serve'.")

# Display processing mode
if use_model:
    st.info("Running on Llama 3.2 1B LLM via Ollama")
else:
    st.info("Running on spaCy (LLM not available)")

# Progress bar
my_bar = st.progress(0, text="Awaiting resume upload...")

# Common technical skills
COMMON_SKILLS = {
    "python", "java", "javascript", "c++", "c#", "sql", "html", "css", "react", "angular",
    "node.js", "django", "flask", "spring", "mysql", "postgresql", "mongodb", "aws",
    "azure", "gcp", "docker", "kubernetes", "git", "jenkins", "tensorflow", "pytorch",
    "web scraping", "api", "rest", "graphql", "linux", "bash", "agile", "scrum"
}

# Course recommendations for skills
COURSE_RECOMMENDATIONS = {
    "python": ["Python for Everybody (Coursera)", "Learn Python 3 (Udemy)"],
    "java": ["Java Programming Masterclass (Udemy)", "Java Programming (Coursera)"],
    "javascript": ["JavaScript: The Complete Guide (Udemy)", "Eloquent JavaScript (Free)"],
    "sql": ["SQL for Data Science (Coursera)", "The Complete SQL Bootcamp (Udemy)"],
    "react": ["React - The Complete Guide (Udemy)", "React JS (Coursera)"],
    "aws": ["AWS Certified Solutions Architect (Udemy)", "AWS Fundamentals (Coursera)"],
    "docker": ["Docker Mastery (Udemy)", "Docker for DevOps (Coursera)"],
    "tensorflow": ["Deep Learning with TensorFlow (Coursera)", "TensorFlow Developer Certificate (Udemy)"],
    "web scraping": ["Web Scraping with Python (Udemy)", "Data Scraping with BeautifulSoup (Coursera)"],
    "cloud computing": ["Cloud Computing Basics (Coursera)", "Google Cloud Professional (Udemy)"]
}

def generate_text(prompt, max_length=50):
    """Generate text using Llama 3.2 1B."""
    if not use_model:
        return ""
    try:
        response = ollama.generate(model="llama3.2:1b", prompt=prompt, options={"num_tokens": max_length})
        return response['response'].strip()
    except Exception as e:
        st.error(f"Text generation failed: {e}")
        return ""

def serialize_resume(resume_data):
    """Convert resume to string for LLM."""
    sections = []
    if "skills" in resume_data and resume_data["skills"]:
        sections.append("Skills: " + ", ".join(resume_data["skills"]))
    if "education" in resume_data and resume_data["education"]:
        sections.append("Education: " + resume_data["education"])
    if "experience" in resume_data and resume_data["experience"]:
        sections.append("Experience: " + "; ".join(resume_data["experience"]))
    if "projects" in resume_data and resume_data["projects"]:
        sections.append("Projects: " + "; ".join(resume_data["projects"]))
    return "\n".join(sections)

def serialize_job(job):
    """Convert job to string for LLM."""
    return f"Job Title: {job['name']}\nCompany: {job['company']}\nLocation: {job['location']}\nApply Link: {job['apply_link']}"

def check_resume(resume):
    """Validate resume and extract text."""
    try:
        if resume is None:
            st.error("No resume file provided")
            return False, ""
        with pdfplumber.open(resume) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        tokens = nltk.word_tokenize(text)
        total_length = sum(len(token) for token in tokens)
        if total_length >= 16000:
            st.warning("Resume is too long (exceeds 16,000 tokens)")
            return False, text
        required_sections = ["summary", "skills", "experience", "projects", "education"]
        text_lower = text.lower()
        has_sections = any(section in text_lower for section in required_sections)
        if not has_sections:
            st.warning("Resume lacks required sections")
            return False, text
        st.write("Debug: Resume validated successfully")
        return True, text
    except Exception as e:
        st.error(f"Resume validation failed: {e}")
        return False, ""

def resume_into_json(resume_file):
    """Parse resume into structured data."""
    try:
        with pdfplumber.open(resume_file) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
            table_text = []
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        table_text.append(" ".join(cell or "" for cell in row))
            text += "\n" + "\n".join(table_text)
        st.write("Debug: Raw resume text (first 500 chars):", text[:500])
        resume_data = {"skills": [], "experience": [], "projects": [], "education": ""}
        lines = text.split("\n")
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Detect section headers
            if re.match(r"^(skills|technical skills|expertise|proficiencies|technical proficiencies|core competencies|abilities|competencies|key skills)$", line.lower()):
                current_section = "skills"
            elif re.match(r"^(education|academic|academics|qualifications)$", line.lower()):
                current_section = "education"
            elif re.match(r"^(experience|work experience|employment|professional experience|work history)$", line.lower()):
                current_section = "experience"
            elif re.match(r"^(projects|portfolio|personal projects|project experience)$", line.lower()):
                current_section = "projects"
            elif current_section == "skills":
                skills = [s.strip() for s in re.split(r"[,;•\-\t\n|:]+", line) if s.strip() and len(s.strip()) > 2]
                resume_data["skills"].extend(skills)
            elif current_section == "education":
                resume_data["education"] += line + "; "
            elif current_section == "experience":
                resume_data["experience"].append(line)
            elif current_section == "projects":
                resume_data["projects"].append(line)
        # LLM-based skill detection
        if use_model:
            prompt = f"Extract technical skills from this resume text:\n\n{text[:2000]}\n\nSkills (comma-separated):"
            llm_skills = generate_text(prompt, max_length=100)
            if llm_skills:
                resume_data["skills"].extend([s.strip() for s in llm_skills.split(",") if s.strip()])
        # Keyword-based detection
        text_lower = text.lower()
        for skill in COMMON_SKILLS:
            if skill in text_lower and skill not in resume_data["skills"]:
                resume_data["skills"].append(skill)
        # Clean up
        resume_data["skills"] = list(set(s.lower().strip() for s in resume_data["skills"] if len(s.strip()) > 2))
        resume_data["experience"] = list(set(resume_data["experience"]))
        resume_data["projects"] = list(set(resume_data["projects"]))
        resume_data["education"] = resume_data["education"].strip("; ")
        st.write("Debug: Extracted skills:", resume_data["skills"])
        return resume_data
    except Exception as e:
        st.error(f"Resume parsing failed: {e}")
        return {}

def generate_query_spacy(resume_data):
    """Generate search query using spaCy."""
    if nlp is None:
        st.warning("spaCy not available. Using default query.")
        return "internship"
    try:
        text = " ".join(
            resume_data.get("skills", []) +
            resume_data.get("experience", []) +
            resume_data.get("projects", []) +
            [resume_data.get("education", "")]
        )
        doc = nlp(text)
        keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2]
        keywords = list(set(keywords))[:5]
        query = "internship " + " ".join(keywords)
        return query if keywords else "internship"
    except Exception as e:
        st.error(f"spaCy query generation failed: {e}")
        return "internship"

def search_database(query):
    """Fetch internships from SerpAPI engines."""
    try:
        api_key = "6817e94b9eed55dea56565370a36742923b1d3dbffa1af7a2293eed639a7b390"
        engines = ["google_jobs", "bing", "yahoo"]
        jobs = []
        for engine in engines:
            st.write(f"Debug: Querying {engine} with query: {query}")
            params = {
                "engine": engine,
                "q": f"{query} internship",
                "hl": "en",
                "api_key": api_key,
                "num": 5
            }
            search = GoogleSearch(params)
            if engine == "google_jobs":
                results = search.get_dict().get("jobs_results", [])
                for result in results:
                    jobs.append({
                        "name": result.get("title", "Unknown"),
                        "company": result.get("company_name", "Unknown"),
                        "location": result.get("location", "Unknown"),
                        "apply_link": result.get("related_links", [{}])[0].get("link", "https://example.com"),
                        "description": result.get("description", ""),
                        "source": engine
                    })
            else:
                results = search.get_dict().get("organic_results", [])
                for result in results:
                    jobs.append({
                        "name": result.get("title", "Unknown"),
                        "company": result.get("source", "Unknown"),
                        "location": "Unknown",
                        "apply_link": result.get("link", "https://example.com"),
                        "description": result.get("snippet", ""),
                        "source": engine
                    })
        st.write("Debug: Fetched jobs:", [job["name"] for job in jobs])
        return jobs[:10]
    except Exception as e:
        st.error(f"SerpAPI search failed: {e}")
        return []

def deduplicate(passages):
    """Remove duplicate jobs."""
    seen = set()
    unique_jobs = []
    for passage in passages:
        for job in passage:
            job_tuple = (job["name"].lower(), job["company"].lower())
            if job_tuple not in seen:
                seen.add(job_tuple)
                unique_jobs.append(job)
    return unique_jobs

def analyze_job_fit(resume_data, job):
    """Analyze job fit using cosine similarity."""
    resume_skills = resume_data.get("skills", [])
    job_description = job.get("description", "")
    if use_model:
        try:
            prompt = f"Extract required skills from this job description:\n\n{job_description}\n\nRequired skills (comma-separated):"
            required_skills_text = generate_text(prompt, max_length=100)
            required_skills = [s.strip().lower() for s in required_skills_text.split(",") if s.strip()]
        except Exception as e:
            st.error(f"Skill extraction failed: {e}")
            required_skills = []
    else:
        if nlp:
            doc = nlp(job_description)
            required_skills = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2]
            required_skills = list(set(required_skills))[:10]
        else:
            required_skills = []
    # Cosine similarity
    if resume_skills and required_skills:
        vectorizer = TfidfVectorizer()
        skill_vectors = vectorizer.fit_transform([" ".join(resume_skills), " ".join(required_skills)])
        fit_percentage = cosine_similarity(skill_vectors[0], skill_vectors[1])[0][0] * 100
        matching_skills = [s for s in resume_skills if any(s.lower() in r.lower() for r in required_skills)]
        missing_skills = [s for s in required_skills if not any(s.lower() in r.lower() for r in resume_skills)]
    else:
        fit_percentage = 0
        matching_skills = []
        missing_skills = required_skills
    fit_percentage = min(max(fit_percentage, 0), 100)
    st.write(f"Debug: Job '{job['name']}': Matching skills={matching_skills}, Missing skills={missing_skills}, Fit={fit_percentage}%")
    return {
        "fit_percentage": round(fit_percentage, 2),
        "matching_skills": matching_skills,
        "missing_skills": missing_skills
    }

def create_pie_chart(fit_percentage, job_title):
    """Create a pie chart for job fit."""
    try:
        if not (0 <= fit_percentage <= 100):
            st.warning(f"Invalid fit percentage ({fit_percentage}) for {job_title}. Using 0%.")
            fit_percentage = 0
        labels = ['Matching Skills', 'Missing Skills']
        sizes = [fit_percentage, 100 - fit_percentage]
        colors = ['#66b3ff', '#ff9999']
        plt.figure(figsize=(4, 4))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f"Fit for {job_title[:30]}...")
        safe_title = "".join(c for c in job_title if c.isalnum() or c in " _")[:30]
        chart_path = f"pie_chart_{safe_title}.png"
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        st.write(f"Debug: Pie chart saved at {chart_path}")
        return chart_path
    except Exception as e:
        st.error(f"Pie chart creation failed for {job_title}: {e}")
        return None

def get_course_recommendations(missing_skills):
    """Suggest courses for missing skills."""
    courses = []
    for skill in missing_skills:
        skill_lower = skill.lower()
        for key, course_list in COURSE_RECOMMENDATIONS.items():
            if key in skill_lower:
                courses.extend(course_list)
        if not courses:
            courses.append(f"Search for '{skill}' on Coursera or Udemy")
    return list(set(courses))[:3]

def find_internships(resume_data, resume_text):
    """Find and analyze internships."""
    passages = []
    resume_serialized = serialize_resume(resume_data)
    for hop in range(3):
        if use_model:
            prompt = f"Generate a search query for internships based on this resume skills: {', '.join(resume_data.get('skills', []))}\n\nSearch query:"
            query = generate_text(prompt, max_length=50)
            if not query:
                query = generate_query_spacy(resume_data)
        else:
            query = generate_query_spacy(resume_data)
        st.write(f"Debug: Query {hop+1}: {query}")
        info = search_database(query)
        passages.append(info)
    context = deduplicate(passages)
    my_bar.progress(60, text="Analyzing internships...")
    analyzed_jobs = []
    for job in context[:5]:
        job_serialized = serialize_job(job)
        fit_analysis = analyze_job_fit(resume_data, job)
        job["fit_percentage"] = fit_analysis["fit_percentage"]
        job["matching_skills"] = fit_analysis["matching_skills"]
        job["missing_skills"] = fit_analysis["missing_skills"]
        job["course_recommendations"] = get_course_recommendations(job["missing_skills"])
        if use_model:
            prompt = f"Analyze how well this resume matches the internship and suggest improvements:\n\nResume:\n{resume_serialized}\n\nInternship:\n{job_serialized}\n\nMatch analysis and tips:"
            analysis = generate_text(prompt, max_length=150)
            job["match_analysis"] = analysis if analysis else "No analysis available"
        else:
            job["match_analysis"] = "No analysis available (LLM not active)"
        chart_path = create_pie_chart(job["fit_percentage"], job["name"])
        job["pie_chart"] = chart_path
        analyzed_jobs.append(job)
    for job in context[5:]:
        job["match_analysis"] = "Analysis not performed (limit reached)"
        job["fit_percentage"] = 0
        job["matching_skills"] = []
        job["missing_skills"] = []
        job["course_recommendations"] = []
        job["pie_chart"] = None
        analyzed_jobs.append(job)
    # Sort by fit percentage
    analyzed_jobs.sort(key=lambda x: x["fit_percentage"], reverse=True)
    return json.dumps(analyzed_jobs)

def company_url(company_name):
    """Generate mock company URL."""
    return f"https://{company_name.lower().replace(' ', '')}.com"

def main():
    """Main Streamlit app."""
    st.title("Internship Finder")
    st.markdown("""
        <style>
        .resume-section {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
        }
        .section-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .skills-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .skill-tag {
            background-color: #e0e7ff;
            color: #333;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        .list-item {
            margin-bottom: 5px;
            font-size: 0.95em;
        }
        .best-match {
            background-color: #d4edda;
            padding: 5px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    file = st.file_uploader("Upload Resume to get started", type=["pdf"], help="Upload a PDF resume (max 200MB)")
    if file is not None:
        st.write("Debug: Resume uploaded")
        is_valid, resume_text = check_resume(file)
        if is_valid:
            with st.status("Extracting Details from Resume"):
                resume_data = resume_into_json(file)
                st.subheader("Extracted Resume Details")
                with st.container():
                    st.markdown('<div class="resume-section"><div class="section-title">Skills</div>', unsafe_allow_html=True)
                    if resume_data.get("skills", []):
                        st.markdown('<div class="skills-list">', unsafe_allow_html=True)
                        for skill in resume_data["skills"]:
                            st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="list-item">None</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="resume-section"><div class="section-title">Education</div>', unsafe_allow_html=True)
                    if resume_data.get("education", ""):
                        st.markdown(f'<div class="list-item">{resume_data["education"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="list-item">None</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="resume-section"><div class="section-title">Experience</div>', unsafe_allow_html=True)
                    if resume_data.get("experience", []):
                        for i, exp in enumerate(resume_data["experience"], 1):
                            st.markdown(f'<div class="list-item">{i}. {exp}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="list-item">None</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="resume-section"><div class="section-title">Projects</div>', unsafe_allow_html=True)
                    if resume_data.get("projects", []):
                        for i, proj in enumerate(resume_data["projects"], 1):
                            st.markdown(f'<div class="list-item">{i}. {proj}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="list-item">None</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            with st.spinner("Searching for internships..."):
                generate = find_internships(resume_data, resume_text)
            if generate:
                st.subheader("List of Internships (Best Matches First)")
                try:
                    interns = json.loads(generate)
                    my_bar.progress(100, "Internships Found!")
                    for i, intern in enumerate(interns):
                        title = f"{intern['name']} at {intern['company']}"
                        if i == 0 and intern['fit_percentage'] > 50:
                            title = f"🌟 Best Match: {title}"
                            st.markdown('<div class="best-match">', unsafe_allow_html=True)
                        with st.expander(title):
                            st.write(f"*Location*: {intern['location']}")
                            st.write(f"*Source*: {intern['source']}")
                            st.link_button("Apply", intern["apply_link"])
                            st.write(f"*Match Analysis*: {intern.get('match_analysis', 'No analysis available')}")
                            st.write(f"*Fit Percentage*: {intern['fit_percentage']}%")
                            if intern.get("pie_chart"):
                                st.image(intern["pie_chart"], caption=f"Fit for {intern['name']}")
                            st.write("*Matching Skills*:")
                            st.write(", ".join(intern["matching_skills"]) or "None")
                            st.write("*Skills to Learn*:")
                            st.write(", ".join(intern["missing_skills"]) or "None")
                            st.write("*Recommended Courses*:")
                            st.write(", ".join(intern["course_recommendations"]) or "None")
                        if i == 0 and intern['fit_percentage'] > 50:
                            st.markdown('</div>', unsafe_allow_html=True)
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse internship data: {e}")
                    st.write("Raw data:", generate)
            else:
                my_bar.progress(100, "No Internships Found")
                st.warning("Sorry, no internships found. Please try again later.")
        else:
            st.error("Invalid File Uploaded!")
            my_bar.progress(0, text="Invalid File Uploaded")
    else:
        my_bar.progress(0, text="Awaiting resume upload...")

if _name_ == "_main_":
    main()