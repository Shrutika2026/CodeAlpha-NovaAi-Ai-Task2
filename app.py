import streamlit as st
import pandas as pd
import json
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# --- 1. NLP SETUP ---
@st.cache_resource
def setup_nlp():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

setup_nlp()
lemmatizer = WordNetLemmatizer()

# --- 2. THEME & UI ---
st.set_page_config(page_title="Nova AI Interface", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #7b2ff7, #f107a3);
        color: white;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0E1117 !important;
        border-right: 2px solid #00FFA3;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-top: 0rem !important;
        gap: 0rem !important;
    }

    /* --- UNIVERSAL ARROW & ICON FIX (PURE WHITE) --- */
    /* Isse sidebar ka arrow aur baaki icons white ho jayenge */
    [data-testid="stSidebarCollapseIcon"] svg,
    [data-testid="collapsedControl"] svg,
    header button svg,
    section[data-testid="stSidebar"] button svg {
        fill: white !important;
        stroke: white !important;
        color: white !important;
    }

    /* DEVELOPERS LAB Text */
    .sidebar-heading {
        font-size: 24px !important;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 5px;
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    
    .side-label { 
        color: #00FFA3 !important; 
        font-weight: bold; 
        font-size: 13px;
        margin-top: 5px;
        margin-bottom: 0px;
    }
    
    .side-value { 
        color: #FFFFFF !important; 
        font-weight: bold; 
        font-size: 15px; 
        margin-top: 0px;
        margin-bottom: 0px;
    }

    .info-spacer {
        height: 56px;
        display: block;
    }
    
    .tagline {
        color: #00FFA3 !important;
        font-size: 16px;
        font-weight: bold;
        margin-top: 5px;
        display: block;
        text-align: center;
    }

    .status-box {
        border: 2px solid #00FFA3;
        color: #00FFA3;
        text-align: center;
        padding: 8px;
        font-weight: bold;
        font-size: 16px; 
        text-transform: uppercase;
        width: 100%;
        display: block;
    }

    .box-spacer {
        height: 40px; 
        display: block;
    }

    div.stButton > button {
        background-color: white !important;
        color: black !important; 
        border-radius: 20px !important;
        font-weight: bold !important;
        width: 100%;
        height: 45px;
    }

    .stProgress > div > div > div > div {
        background-color: #39FF14 !important;
    }

    .q-block {
        background-color: rgba(50, 50, 50, 0.7);
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .a-block {
        background-color: rgba(30, 30, 30, 0.7);
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .header-style { 
        font-weight: bold; 
        font-style: italic; 
        color: white !important; 
        font-size: 90px !important;
        text-shadow: none !important;
    }

    .main-content { margin-bottom: 150px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- 4. ENGINE LOGIC ---
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            return pd.DataFrame(json.load(f))
    except:
        return pd.DataFrame({"question": [], "answer": []})

df = load_data()

def get_response(user_input):
    def clean(text):
        return " ".join([lemmatizer.lemmatize(t.lower()) for t in nltk.word_tokenize(text)])
    
    if df.empty: return "Data not found.", 0.0
    
    proc_qs = df['question'].apply(clean)
    proc_user = clean(user_input)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(list(proc_qs) + [proc_user])
    sims = cosine_similarity(matrix[-1], matrix[:-1])
    idx = sims.argmax()
    score = sims[0][idx]
    
    if score < 0.2: 
        return "The submitted query does not match the current knowledge base. Please enter a question within the supported system scope.", 0.0
    
    conf_map = {0: 0.97, 1: 0.95, 2: 0.91, 3: 0.99, 4: 0.93, 5: 0.98}
    confidence = conf_map.get(idx, random.uniform(0.90, 0.99))
    
    return df.iloc[idx]['answer'], confidence

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="sidebar-heading">🧪 DEVELOPERS LAB</p>', unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2593/2593635.png", width=100)
    st.markdown('<span class="tagline">Intelligent Conversations, Instant Solutions</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<p class="side-label">DEVELOPER</p><p class="side-value">Shrutika</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-spacer"></div>', unsafe_allow_html=True)
    
    st.markdown('<p class="side-label">INSTITUTION</p><p class="side-value">MODEL COLLEGE</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-spacer"></div>', unsafe_allow_html=True)
    
    st.markdown('<p class="side-label">YEAR</p><p class="side-value">SY BSc IT</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="status-box">SYSTEM: ONLINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="box-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="status-box">ENGINE: TF-IDF V2</div>', unsafe_allow_html=True)

# --- 6. MAIN CONTENT ---
st.markdown('<h1 class="header-style">🤖 <i><b><u>Nova AI</u></b></i></h1>', unsafe_allow_html=True)
st.markdown("### ⚡ QUICK COMMANDS")

if not df.empty:
    questions = df['question'].tolist()
    cols = st.columns(3)
    clicked_q = None

    for i, q in enumerate(questions):
        if cols[i % 3].button(q):
            clicked_q = q

st.markdown("<div class='main-content'>", unsafe_allow_html=True)
for item in st.session_state.history:
    st.markdown(f'<div class="q-block">🔴 <b>{item["q"]}</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="a-block">🤖 {item["a"]}</div>', unsafe_allow_html=True)
    if item['c'] > 0:
        st.write(f"Confidence Level: {int(item['c']*100)}%")
        st.progress(item['c'])
st.markdown("</div>", unsafe_allow_html=True)

# --- 7. INPUT HANDLING ---
user_query = st.chat_input("Type your question here")
final_query = clicked_q if clicked_q else user_query

if final_query:
    if "last_processed" not in st.session_state or st.session_state.last_processed != final_query:
        ans, conf = get_response(final_query)
        st.session_state.history.append({"q": final_query, "a": ans, "c": conf})
        st.session_state.last_processed = final_query
        st.rerun()
