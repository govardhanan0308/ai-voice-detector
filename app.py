import streamlit as st
import numpy as np
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from feature_extraction import extract_features

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Voice Detector", layout="wide")

# ---------------- LOAD MODEL ----------------
model = load_model("deepfake_voice_detector.h5")

# ---------------- SESSION ----------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ---------------- RESET ----------------
def reset_app():
    st.session_state.analyzed = False
    st.session_state.result = None
    st.rerun()

# ---------------- CSS ----------------
st.markdown("""
<style>

/* 🌤️ SOFT BACKGROUND (LESS INTENSE) */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #e9edf3 !important;
}

/* 🔝 FIX STREAMLIT HEADER (DEPLOY + MENU BUTTON VISIBILITY) */
header {
    background-color: #e9edf3 !important;
}

[data-testid="stToolbar"] {
    background-color: #e9edf3 !important;
}

/* ICON VISIBILITY */
button[kind="header"] svg {
    fill: #374151 !important;
}

/* TEXT */
html, body, p, span, div {
    color: #1f2937 !important;
}

/* HERO */
.hero {
    text-align: center;
}
.hero h1 {
    font-size: 48px;
    background: linear-gradient(90deg, #2563eb, #16a34a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #6b7280;
    font-size: 16px;
}

/* CARD */
.card {
    background: #ffffff;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #d1d5db;
    margin-bottom: 20px;
    animation: fadeUp 0.5s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* BUTTON */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #2563eb, #22c55e);
    color: white;
    font-weight: bold;
    padding: 12px;
    border: none;
}

/* RESULT */
.result {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    margin: 10px 0 20px 0;
}
.real { color: #16a34a; }
.fake { color: #dc2626; }

/* METRIC BOX */
.metric-box {
    background: #f1f5f9;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #d1d5db;
}

/* SMALL TEXT */
.small {
    text-align: center;
    color: #6b7280;
    font-size: 12px;
}

/* PLACEHOLDER */
.placeholder {
    text-align: center;
    color: #6b7280;
    padding: 30px;
}

/* 🔽 REDUCE TOP SPACE */
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 2rem;
}

/* 🔽 REDUCE HERO EXTRA SPACE */
.hero {
    padding-top: 10px !important;
}

/* 📌 STICKY FOOTER */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    background: #e9edf3;
    color: #6b7280;
    font-size: 12px;
    border-top: 1px solid #d1d5db;
    z-index: 100;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
<h1>🎙️ AI Voice Detector</h1>
<p>Advanced Deepfake Voice Detection using AI</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
left, right = st.columns([1, 1], gap="large")

# ---------------- LEFT ----------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📤 Upload Audio")

    uploaded_file = st.file_uploader("", type=["wav", "mp3"])

    if uploaded_file:
        st.audio(uploaded_file)

    analyze = st.button("🚀 Analyze Voice")

    st.markdown('<div class="small">Supported: WAV, MP3</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
# ---------------- RIGHT ----------------
with right:

    if not st.session_state.analyzed:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("""
        <div class="placeholder">
        <h3>🤖 Ready to Analyze</h3>
        <p>Upload audio and click <b>Analyze Voice</b></p>
        <p>Detects Real vs AI Generated Voice</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        is_fake, fake_prob, real_prob, file_path = st.session_state.result

        st.markdown('<div class="card">', unsafe_allow_html=True)

        label = "FAKE VOICE" if is_fake else "REAL VOICE"
        color = "fake" if is_fake else "real"

        # RESULT
        st.markdown(
            f'<div class="result {color}">{"❌" if is_fake else "✅"} {label}</div>',
            unsafe_allow_html=True
        )

        # CONFIDENCE
        st.subheader("Confidence")

        st.write(f"Fake: {fake_prob:.2f}%")
        st.progress(int(fake_prob))

        st.write(f"Real: {real_prob:.2f}%")
        st.progress(int(real_prob))

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ANALYSIS ----------------
if uploaded_file and analyze:

    with st.spinner("🔍 AI analyzing..."):

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        features = extract_features(file_path)
        features = np.array(features)
        features = features.reshape(1, features.shape[0], 1)

        prediction = model.predict(features)[0][0]

        # YOUR ORIGINAL LOGIC
        is_fake = prediction > 0.5

        fake_prob = float(prediction) * 100
        real_prob = 100 - fake_prob

    st.session_state.result = (is_fake, fake_prob, real_prob, file_path)
    st.session_state.analyzed = True
    st.rerun()

# ---------------- INSIGHTS ----------------
if st.session_state.analyzed:

    _, _, _, file_path = st.session_state.result

    y, sr = librosa.load(file_path, sr=22050)

    duration = len(y)/sr
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Audio Insights")

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-box">⏱ {duration:.2f}s</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box">📈 {peak:.2f}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box">⚡ {rms:.2f}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # VISUALS
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6,3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_axis_off()
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6,3))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S)
        img = librosa.display.specshow(S_db, sr=sr, ax=ax2)
        fig2.colorbar(img, ax=ax2)
        st.pyplot(fig2)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Analyze Another"):
        reset_app()

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">Deepfake Detection • Final Year Project</div>', unsafe_allow_html=True)