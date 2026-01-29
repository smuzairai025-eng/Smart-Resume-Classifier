import streamlit as st
import torch
import fitz  # PyMuPDF
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import time

# ================== App Config ==================
st.set_page_config(page_title="Smart Resume Classifier", page_icon="📄")

# ================== Model Paths ==================
MODEL_NAME = "uzairkhanswatii/Smart-Resume-Classifier"  # HF Hub model repo
LABEL_ENCODER_URL = "https://raw.githubusercontent.com/smuzairai025-eng/Smart-Resume-Classifier/main/label_encoder.json"

# ================== Load Model & Tokenizer ==================
@st.cache_resource
def load_modelv2():
    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Load label encoder from GitHub (JSON instead of pickle)
    response = requests.get(LABEL_ENCODER_URL)
    response.raise_for_status()
    classes = json.loads(response.content.decode("utf-8"))  # list of class names in model order

    # Simple wrapper to mimic LabelEncoder API we use (inverse_transform + classes_)
    class SimpleLabelEncoder:
        def __init__(self, classes_):
            self.classes_ = classes_
        def inverse_transform(self, indices):
            return [self.classes_[i] for i in indices]

    label_encoder = SimpleLabelEncoder(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, label_encoder, device

tokenizer, model, label_encoder, device = load_modelv2()

# ================== Text Preprocessing ==================
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # remove punctuation/numbers/symbols
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text

# ================== PDF Text Extraction ==================
def extract_text_from_pdf(uploaded_file) -> str:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ================== Prediction ==================
def predict_resume(text):
    """
    Returns:
      label (str): top predicted label
      confidence (float): top class probability
      probs (np.ndarray): probabilities for all classes (len = num_labels)
    """
    text = preprocess_text(text)

    encoding = tokenizer(
        text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]  # shape: (num_labels,)

    max_idx = int(np.argmax(probs))
    label = label_encoder.inverse_transform([max_idx])[0]
    confidence = float(probs[max_idx])
    return label, confidence, probs

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

# ================== Sidebar Navigation ==================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home / Classifier", "Insights", "About the Model", "Contact / Feedback"]
)

# Ensure state containers exist
if "last_results" not in st.session_state:
    # List of dicts: {"name": str, "label": str, "confidence": float, "probs": np.ndarray}
    st.session_state["last_results"] = []
if "resume_input" not in st.session_state:
    st.session_state["resume_input"] = ""

# ================== Page 1: Home / Classifier ==================
if page == "Home / Classifier":
    st.title("📄 Smart Resume Classifier")
    st.caption("Upload a resume PDF or paste resume text to classify.")

    option = st.radio("Choose input method:", ["Upload PDF", "Enter Text"])
    texts_to_classify = []

    if option == "Upload PDF":
        uploaded_files = st.file_uploader(
            "Choose PDF files", type="pdf", accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    text = extract_text_from_pdf(uploaded_file)
                    texts_to_classify.append((uploaded_file.name, text))
                except Exception as e:
                    st.error(f"Failed to read {uploaded_file.name}: {e}")

    elif option == "Enter Text":
        # Clear callback
        def clear_text():
            st.session_state["resume_input"] = ""

        user_text = st.text_area("Paste your resume text here:", key="resume_input")
        st.button("Clear Text", on_click=clear_text, key="clear_text_btn")

        if user_text.strip():
            texts_to_classify.append(("Manual Input", user_text))

    # Single Predict button (unique key)
    if st.button("Classify Resume", key="predict_run"):
        if texts_to_classify:
            st.subheader("📌 Classified Resumes")

            st.session_state["last_results"] = []  # reset previous results for Insights

            with st.spinner("Analyzing resumes... ⏳"):
                # small pause to show spinner nicely
                time.sleep(0.4)

                for name, text in texts_to_classify:
                    label, confidence, probs = predict_resume(text)

                    # Save to session for Insights page
                    st.session_state["last_results"].append({
                        "name": name,
                        "label": label,
                        "confidence": confidence,
                        "probs": probs,
                    })

                    # Result card
                    st.markdown(
                        f"""
                        <div style="padding:15px; border-radius:12px; box-shadow:0px 2px 6px rgba(0,0,0,0.1); margin-bottom:15px;">
                          <h4 style="margin:0 0 8px 0;">{name}</h4>
                          <b>🏷️ Category:</b> {label}<br>
                          <b>📊 Confidence:</b> {confidence:.2%}<br>
                          <b>🔮 Recommended Roles:</b> {", ".join(recommend_job_roles(label))}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("⚠️ Please upload a PDF or enter resume text before classifying.")

# ================== Page 2: Insights / Analysis ==================
elif page == "Insights":
    st.title("📊 Resume Insights & Analysis")

    if st.session_state["last_results"]:
        # Use the most recent item for the chart (or aggregate—keeping simple)
        latest = st.session_state["last_results"][-1]
        probs = latest["probs"]
        classes = label_encoder.classes_

        # Get top-5 classes for display
        top_k = min(5, len(classes))
        top_idx = np.argsort(probs)[::-1][:top_k]
        top_labels = [classes[i] for i in top_idx]
        top_scores = [probs[i] for i in top_idx]

        st.subheader(f"Top {top_k} Categories for: {latest['name']}")
        fig, ax = plt.subplots()
        ax.bar(top_labels, top_scores)
        ax.set_ylabel("Confidence Score")
        ax.set_title("Confidence Scores by Category")
        plt.xticks(rotation=25, ha="right")
        st.pyplot(fig)

        st.subheader("Extracted Keywords / Skills (Placeholder)")
        st.write("Python, Machine Learning, SQL, Data Analysis, REST APIs, Cloud, Git")
    else:
        st.info("Run a classification first to see insights here.")

# ================== Page 3: About the Model ==================
elif page == "About the Model":
    st.title("ℹ️ About the Smart Resume Classifier")

    st.markdown(
        """
**What it does**  
The Smart Resume Classifier analyzes resume text and predicts the most relevant job category with a confidence score, and suggests related roles.

**How it works (high level)**  
- Preprocesses and normalizes the resume text  
- Uses a Transformer model to generate class probabilities  
- Picks the top category and shows confidence + role recommendations  

**Dataset (high level)**  
- Trained on a curated labeled resume dataset spanning multiple domains (tech, business, engineering, etc.)  

**How to Use**  
1. Upload a **PDF** resume **or** paste text.  
2. Click **Classify Resume**.  
3. View **Category**, **Confidence**, and **Recommended Roles**.  

**Limitations**  
- Works best on text-based PDFs (not scanned images).  
- Confidence depends on the quality/coverage of the data.  
        """
    )

# ================== Page 4: Contact / Feedback ==================
elif page == "Contact / Feedback":
    st.title("📬 Contact & Feedback")

#     st.markdown(
#         """
# Built by **Uzair Khan — ML Engineer**  

# 📧 Email: Uzairkhan242002@gmail.com
#         """
#     )

    st.subheader("Feedback Form")
    feedback = st.text_area("Share your feedback or suggestions:")
    if st.button("Submit Feedback", key="feedback_btn"):
        if feedback.strip():
            st.success("✅ Thank you for your feedback!")
            # (Optional) Save to a file or a DB here.
        else:
            st.warning("⚠️ Please enter some feedback before submitting.")

# ================== Footer ==================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<a href='https://github.com/smuzairai025-eng'>GitHub</a></center>",
    unsafe_allow_html=True,
)
# "<center>Built by Uzair Khan — ML Engineer | 🔗 "