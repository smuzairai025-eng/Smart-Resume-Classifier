import streamlit as st
import torch
import fitz  # PyMuPDF
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import io
import json
import re

# ================== Load Model & Tokenizer ==================
st.set_page_config(page_title="Smart Resume Classifier", page_icon="📄")

MODEL_NAME = "uzairkhanswatii/Smart-Resume-Classifier"  # HF Hub model repo
LABEL_ENCODER_URL = "https://raw.githubusercontent.com/Uzairkhanswatii/Smart-Resume-Classifier/main/label_encoder.json"

def load_modelv2():
    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Load label encoder from GitHub (JSON instead of pickle)
    response = requests.get(LABEL_ENCODER_URL)
    response.raise_for_status()
    classes = json.loads(response.content.decode("utf-8"))

    # Custom "encoder" wrapper to mimic LabelEncoder API
    class SimpleLabelEncoder:
        def __init__(self, classes):
            self.classes_ = classes
        def inverse_transform(self, indices):
            return [self.classes_[i] for i in indices]

    label_encoder = SimpleLabelEncoder(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, label_encoder, device


# Call once and cache
tokenizer, model, label_encoder, device = load_modelv2()

# ================== Text Preprocessing ==================
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text

# ================== PDF Text Extraction ==================
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ================== Prediction ==================
def predict_resume(text):
    text = preprocess_text(text)

    encoding = tokenizer(
        text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    max_idx = probs.argmax()
    label = label_encoder.inverse_transform([max_idx])[0]
    confidence = probs[max_idx]
    return label, confidence

# ================== Recommendation Engine ==================
def recommend_job_roles(predicted_label):
    recommendations = {
        "Accountant": ["Financial Analyst", "Auditor", "Tax Consultant"],
        "Advocate": ["Legal Advisor", "Corporate Lawyer", "Paralegal"],
        "Agriculture": ["Farm Manager", "Agronomist", "Agricultural Scientist"],
        "Apparel": ["Fashion Designer", "Merchandiser", "Textile Engineer"],
        "Architecture": ["Interior Designer", "Urban Planner", "Landscape Architect"],
        "Arts": ["Graphic Designer", "Animator", "Creative Director"],
        "Automobile": ["Automotive Engineer", "Vehicle Designer", "Production Engineer"],
        "Aviation": ["Pilot", "Aircraft Maintenance Engineer", "Aerospace Engineer"],
        "BPO": ["Customer Support", "Call Center Executive", "Team Lead"],
        "Banking": ["Credit Analyst", "Investment Banker", "Loan Officer"],
        "Blockchain": ["Smart Contract Developer", "Crypto Analyst", "Blockchain Architect"],
        "Building and Construction": ["Site Engineer", "Quantity Surveyor", "Project Engineer"],
        "Business Analyst": ["Product Analyst", "Data Analyst", "Strategy Consultant"],
        "Civil Engineer": ["Structural Engineer", "Construction Manager", "Geotechnical Engineer"],
        "Consultant": ["Management Consultant", "IT Consultant", "Strategy Advisor"],
        "Data Science": ["Data Analyst", "ML Engineer", "AI Researcher"],
        "Database": ["Database Administrator", "SQL Developer", "Data Architect"],
        "Designing": ["UI/UX Designer", "Graphic Designer", "Product Designer"],
        "DevOps": ["Cloud Engineer", "CI/CD Engineer", "Site Reliability Engineer"],
        "Digital Media": ["SEO Specialist", "Content Manager", "Social Media Strategist"],
        "DotNet Developer": [".NET Core Developer", "Backend Engineer", "C# Developer"],
        "ETL Developer": ["Data Engineer", "Informatica Developer", "BI Developer"],
        "Education": ["Teacher", "Curriculum Developer", "Academic Coordinator"],
        "Electrical Engineering": ["Power Engineer", "Electronics Engineer", "Control Systems Engineer"],
        "Finance": ["Investment Analyst", "Portfolio Manager", "Financial Controller"],
        "Food and Beverages": ["Food Technologist", "Quality Assurance Specialist", "Production Manager"],
        "Health and Fitness": ["Dietician", "Personal Trainer", "Health Coach"],
        "Human Resources": ["Talent Acquisition Specialist", "HR Generalist", "Compensation Analyst"],
        "Information Technology": ["System Administrator", "IT Support Specialist", "Solutions Architect"],
        "Java Developer": ["Spring Boot Developer", "Backend Engineer", "J2EE Developer"],
        "Management": ["Operations Manager", "Business Manager", "Program Manager"],
        "Mechanical Engineer": ["Design Engineer", "Manufacturing Engineer", "HVAC Engineer"],
        "Network Security Engineer": ["Cybersecurity Analyst", "SOC Analyst", "Security Consultant"],
        "Operations Manager": ["Logistics Manager", "Production Manager", "Supply Chain Manager"],
        "PMO": ["Project Coordinator", "Program Manager", "Portfolio Analyst"],
        "Public Relations": ["Media Relations Specialist", "Communications Manager", "Brand Strategist"],
        "Python Developer": ["Backend Engineer", "Django Developer", "AI Engineer"],
        "React Developer": ["Frontend Developer", "Fullstack Developer", "UI Engineer"],
        "SAP Developer": ["SAP ABAP Consultant", "SAP Functional Consultant", "ERP Specialist"],
        "SQL Developer": ["Database Engineer", "ETL Developer", "BI Specialist"],
        "Sales": ["Sales Executive", "Account Manager", "Business Development Manager"],
        "Testing": ["QA Engineer", "Automation Tester", "SDET"],
        "Web Designing": ["UI Designer", "Frontend Developer", "Visual Designer"],
    }
    return recommendations.get(predicted_label, ["No recommendations available"])


# ================== Streamlit UI ==================
st.title("📄 Smart Resume Classifier")
st.caption("Upload a resume PDF or paste resume text to classify.")

# Input options
option = st.radio("Choose input method:", ["Upload PDF", "Enter Text"])

texts_to_classify = []

if option == "Upload PDF":
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            texts_to_classify.append((uploaded_file.name, text))

elif option == "Enter Text":
    # Define a clear function for session state
    def clear_text():
        st.session_state["resume_input"] = ""

    # Text area bound to session_state
    user_text = st.text_area("Paste your resume text here:", key="resume_input")

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("Clear Text", on_click=clear_text)
    with col2:
        predict_clicked = st.button("Predict")

    # Only append text when Predict is clicked
    if predict_clicked and user_text.strip():
        texts_to_classify.append(("Manual Input", user_text))

# Predict button
if st.button("Predict"):
    if texts_to_classify:
        st.subheader("📌 Classified Resumes")
        with st.container():
            for name, text in texts_to_classify:
                label, confidence = predict_resume(text)
                st.markdown(f"**{name}** → 🏷️ {label} (Confidence: {confidence:.2f})")

                # Show recommendations
                recs = recommend_job_roles(label)
                st.write("🔮 Recommended Roles:", ", ".join(recs))
                st.divider()

