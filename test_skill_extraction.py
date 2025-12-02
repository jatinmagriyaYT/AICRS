import os
import django
import re

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Career_Recommender.settings')
django.setup()

from ai_recommender.services import extract_skills_from_text, clean_skill_list

def test_extraction():
    print("--- TESTING SKILL EXTRACTION ---")
    
    # Text provided by user (simulating resume content)
    resume_text = """
    Technical Skills (Hard Skills)
    • Programming Languages: Python
    • Frameworks (Backend): Django, Flask, FastAPI, Flet (Python Flutter), Tkinter
    • Frontend: HTML, CSS, Bootstrap, JavaScript
    • Databases: SQLite3, SQLAlchemy
    • Version Control: Git & GitHub
    • Other Tools: Automation, ChatGPT, DeepSeek, Canva, Cloud Platforms (Google Cloud, Microsoft Azure)
    
    Soft Skills
    • Leadership, Public Speaking, Collaboration, Problem-solving, Analytical Thinking, Time Management, Adaptability
    """
    
    print("Input Text (Snippet):")
    print(resume_text.strip())
    print("-" * 30)
    
    # Run extraction
    extracted_skills = extract_skills_from_text(resume_text)
    
    print(f"\nExtracted {len(extracted_skills)} skills:")
    print(", ".join(extracted_skills))
    
    # Expected skills
    expected = {
        'python', 'django', 'flask', 'fastapi', 'flet', 'tkinter', 
        'html', 'css', 'bootstrap', 'javascript', 'sqlite3', 'sqlalchemy',
        'git', 'github', 'google cloud', 'microsoft azure'
    }
    
    extracted_lower = {s.lower() for s in extracted_skills}
    
    missing = expected - extracted_lower
    print(f"\nMissing Expected Skills ({len(missing)}):")
    print(", ".join(missing))
    
    # Check for false positives or weird formatting
    print("\nPotential Issues:")
    for s in extracted_skills:
        if len(s) < 2 and s.lower() not in ['c', 'r']:
            print(f"  - Too short: '{s}'")
        if ' ' in s and len(s.split()) > 3:
            print(f"  - Too long: '{s}'")

if __name__ == "__main__":
    test_extraction()
