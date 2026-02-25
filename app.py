import streamlit as st
import pickle
import nltk
import string
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

def text_transform(message):
    message = message.lower()
    message = nltk.word_tokenize(message)
    y = []
    for i in message:
        if i.isalnum():
            y.append(i)
        y.clear()
    for i in message:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    message = y[:]
    y.clear()
    for i in message:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.set_page_config(page_title="SpamShield AI", page_icon="🛡️", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
html, body, .stApp { background-color: #080b14 !important; color: #e8eaf0; font-family: "Syne", sans-serif; }
#MainMenu, footer, header { display: none !important; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 760px !important; }
.stApp::before { content:""; position:fixed; inset:0; background-image:linear-gradient(rgba(0,255,180,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,180,0.025) 1px,transparent 1px); background-size:44px 44px; pointer-events:none; z-index:0; animation:gridPulse 6s ease-in-out infinite; }
@keyframes gridPulse { 0%,100%{opacity:.4} 50%{opacity:1} }
@keyframes fadeDown { from{opacity:0;transform:translateY(-18px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeUp   { from{opacity:0;transform:translateY(14px)}  to{opacity:1;transform:translateY(0)} }
@keyframes resultPop { from{opacity:0;transform:scale(.82)} to{opacity:1;transform:scale(1)} }
.hero { text-align:center; padding:3rem 0 1.5rem; }
.hero-badge { display:inline-block; background:rgba(0,255,180,0.08); border:1px solid rgba(0,255,180,0.25); color:#00ffb4; font-family:"DM Mono",monospace; font-size:.68rem; letter-spacing:.2em; padding:5px 16px; border-radius:20px; margin-bottom:1.4rem; animation:fadeDown .5s ease forwards; }
.hero-title { font-size:clamp(2.6rem,8vw,4.2rem); font-weight:800; line-height:1.05; letter-spacing:-.03em; margin-bottom:.8rem; animation:fadeDown .6s ease forwards; }
.hero-title .green { color:#00ffb4; } .hero-title .dim { color:#1e2540; }
.hero-sub { font-family:"DM Mono",monospace; font-size:.8rem; color:#3a4060; letter-spacing:.08em; }
.stats-row { display:flex; gap:.75rem; margin:2rem 0 1.5rem; }
.stat-card { flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:.9rem .5rem; text-align:center; transition:border-color .3s,background .3s,transform .3s; }
.stat-card:hover { border-color:rgba(0,255,180,0.3); background:rgba(0,255,180,0.04); transform:translateY(-3px); }
.stat-value { font-size:1.5rem; font-weight:800; color:#00ffb4; }
.stat-label { font-family:"DM Mono",monospace; font-size:.6rem; color:#3a4060; letter-spacing:.12em; text-transform:uppercase; margin-top:3px; }
.input-panel { background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07); border-radius:20px; padding:1.8rem 1.8rem 1.2rem; margin:0 0 1rem; transition:border-color .3s; }
.input-panel:hover { border-color:rgba(0,255,180,0.15); }
.input-label { font-family:"DM Mono",monospace; font-size:.68rem; letter-spacing:.15em; color:#00ffb4; text-transform:uppercase; margin-bottom:.6rem; }
.stTextArea textarea { background:rgba(0,0,0,0.5) !important; border:1px solid rgba(255,255,255,0.08) !important; border-radius:12px !important; color:#e8eaf0 !important; font-family:"DM Mono",monospace !important; font-size:.88rem !important; line-height:1.75 !important; padding:1rem !important; caret-color:#00ffb4 !important; }
.stTextArea textarea:focus { border-color:rgba(0,255,180,0.45) !important; box-shadow:0 0 0 3px rgba(0,255,180,0.07) !important; }
.stTextArea textarea::placeholder { color:#2a3050 !important; }
.stTextArea label { display:none !important; }
.counter-bar { display:flex; justify-content:space-between; margin-top:.4rem; font-family:"DM Mono",monospace; font-size:.7rem; color:#2a3050; }
.counter-bar .active { color:#00ffb4; } .counter-bar .warn { color:#ffb400; }
.stButton > button { background:linear-gradient(135deg,#00ffb4,#00d4ff) !important; color:#080b14 !important; font-family:"Syne",sans-serif !important; font-weight:700 !important; font-size:.95rem !important; border:none !important; border-radius:12px !important; padding:.8rem 2rem !important; width:100% !important; transition:all .25s ease !important; box-shadow:0 4px 20px rgba(0,255,180,0.2) !important; }
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 28px rgba(0,255,180,0.35) !important; }
.result-card { border-radius:20px; padding:2.2rem 2rem; text-align:center; margin:1.5rem 0; animation:resultPop .5s cubic-bezier(.34,1.56,.64,1) forwards; position:relative; overflow:hidden; }
.result-spam { background:linear-gradient(135deg,rgba(255,50,50,.13),rgba(255,0,90,.08)); border:1px solid rgba(255,50,50,.35); color:#ff4040; }
.result-ham  { background:linear-gradient(135deg,rgba(0,255,180,.1),rgba(0,210,255,.06));  border:1px solid rgba(0,255,180,.35); color:#00ffb4; }
.result-icon { font-size:2.8rem; margin-bottom:.4rem; }
.result-title { font-size:2.4rem; font-weight:800; letter-spacing:-.02em; margin-bottom:.4rem; }
.result-desc { font-family:"DM Mono",monospace; font-size:.75rem; opacity:.6; letter-spacing:.05em; }
.conf-wrap { margin:.9rem auto 0; max-width:280px; }
.conf-label { font-family:"DM Mono",monospace; font-size:.65rem; color:#3a4060; text-align:left; margin-bottom:5px; letter-spacing:.1em; }
.conf-bg { background:rgba(255,255,255,0.06); border-radius:99px; height:5px; overflow:hidden; }
.conf-fill { height:100%; border-radius:99px; }
.warn-box { background:rgba(255,180,0,.07); border:1px solid rgba(255,180,0,.25); border-radius:12px; padding:.9rem 1.2rem; font-family:"DM Mono",monospace; font-size:.8rem; color:#ffb400; margin:1rem 0; }
.section-label { font-family:"DM Mono",monospace; font-size:.68rem; letter-spacing:.15em; color:#2a3050; text-transform:uppercase; margin:1.8rem 0 .7rem; }
.hist-item { display:flex; align-items:center; gap:.8rem; padding:.7rem 1rem; border-radius:10px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04); margin-bottom:.4rem; font-family:"DM Mono",monospace; font-size:.75rem; transition:border-color .2s; }
.hist-item:hover { border-color:rgba(255,255,255,0.1); }
.hist-badge { font-size:.6rem; font-weight:700; padding:3px 8px; border-radius:6px; white-space:nowrap; letter-spacing:.08em; }
.badge-spam { background:rgba(255,50,50,.12); color:#ff4040; border:1px solid rgba(255,50,50,.25); }
.badge-ham  { background:rgba(0,255,180,.08); color:#00ffb4; border:1px solid rgba(0,255,180,.25); }
.hist-msg { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#2a3050; }
.div { border:none; border-top:1px solid rgba(255,255,255,0.05); margin:1.8rem 0; }
.footer { text-align:center; font-family:"DM Mono",monospace; font-size:.65rem; color:#1e2540; letter-spacing:.1em; padding-top:.5rem; }
.stSpinner > div { border-top-color:#00ffb4 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ MULTINOMIAL NAIVE BAYES · TF-IDF</div>
    <div class="hero-title">Spam<span class="green">Shield</span><br><span class="dim">AI Detector</span></div>
    <div class="hero-sub">// classify &nbsp;·&nbsp; analyse &nbsp;·&nbsp; protect</div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-card"><div class="stat-value">97%</div><div class="stat-label">Accuracy</div></div>
    <div class="stat-card"><div class="stat-value">100%</div><div class="stat-label">Precision</div></div>
    <div class="stat-card"><div class="stat-value">3K</div><div class="stat-label">Features</div></div>
    <div class="stat-card"><div class="stat-value">5.5K</div><div class="stat-label">Training Msgs</div></div>
</div>""", unsafe_allow_html=True)

st.markdown("<div class=\"input-panel\"><div class=\"input-label\">//  Input Message</div>", unsafe_allow_html=True)
input_msg = st.text_area("msg", height=140, placeholder="Paste any SMS or email message here...", value=st.session_state.input_text, key="msg_input")
if input_msg:
    chars, words = len(input_msg), len(input_msg.split())
    wc = "warn" if chars > 300 else "active"
    wt = "⚠️ Very long" if chars > 300 else "✓ Normal"
    st.markdown(f"<div class=\"counter-bar\"><span class=\"active\">{chars} chars · {words} words</span><span class=\"{wc}\">{wt}</span></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

predict_btn = st.button("🛡️ Analyse Message", key="predict")

if predict_btn:
    if not input_msg.strip():
        st.markdown("<div class=\"warn-box\">⚠️ Please enter a message to analyse.</div>", unsafe_allow_html=True)
    else:
        with st.spinner("Scanning..."):
            time.sleep(0.6)
            transformed = text_transform(input_msg)
            vector = tfidf.transform([transformed])
            result = model.predict(vector)[0]
            proba  = model.predict_proba(vector)[0]
            confidence = int(proba[result] * 100)
            st.session_state.result = (result, confidence)
            label   = "SPAM" if result == 1 else "HAM"
            snippet = input_msg[:65] + "..." if len(input_msg) > 65 else input_msg
            st.session_state.history.insert(0, (label, snippet))
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[:5]

if st.session_state.result is not None:
    result, confidence = st.session_state.result
    if result == 1:
        st.markdown(f"""<div class="result-card result-spam"><div class="result-icon">🚨</div><div class="result-title">SPAM</div><div class="result-desc">Suspicious — do not click links or share personal info.</div><div class="conf-wrap"><div class="conf-label">CONFIDENCE · {confidence}%</div><div class="conf-bg"><div class="conf-fill" style="width:{confidence}%;background:linear-gradient(90deg,#ff4040,#ff0064);"></div></div></div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-card result-ham"><div class="result-icon">✅</div><div class="result-title">NOT SPAM</div><div class="result-desc">This message appears safe and legitimate.</div><div class="conf-wrap"><div class="conf-label">CONFIDENCE · {confidence}%</div><div class="conf-bg"><div class="conf-fill" style="width:{confidence}%;background:linear-gradient(90deg,#00ffb4,#00d4ff);"></div></div></div></div>""", unsafe_allow_html=True)

st.markdown("<div class=\"section-label\">//  Try an example</div>", unsafe_allow_html=True)

examples = [
    ("🚨 Spam #1", "WINNER!! You have been selected for a £1000 prize. Call 0800-XXX now!"),
    ("🚨 Spam #2", "FREE entry: text WIN to 83600. You could win £200 cash. T&C apply."),
    ("✅ Ham #1",  "Hey, are we still on for lunch tomorrow at 1pm? Let me know!"),
    ("✅ Ham #2",  "Don't forget to pick up milk and eggs on your way home. Thanks!"),
]
col1, col2 = st.columns(2)
for i, (label, msg) in enumerate(examples):
    with (col1 if i % 2 == 0 else col2):
        if st.button(label, key=f"ex_{i}"):
            transformed = text_transform(msg)
            vector = tfidf.transform([transformed])
            result = model.predict(vector)[0]
            proba  = model.predict_proba(vector)[0]
            confidence = int(proba[result] * 100)
            st.session_state.result = (result, confidence)
            st.session_state.input_text = msg
            snippet = msg[:65] + "..." if len(msg) > 65 else msg
            lbl = "SPAM" if result == 1 else "HAM"
            st.session_state.history.insert(0, (lbl, snippet))
            st.rerun()

if st.session_state.history:
    st.markdown("<hr class=\"div\"><div class=\"section-label\">//  Recent checks</div>", unsafe_allow_html=True)
    for label, snippet in st.session_state.history:
        bc = "badge-spam" if label == "SPAM" else "badge-ham"
        st.markdown(f"<div class=\"hist-item\"><span class=\"hist-badge {bc}\">{label}</span><span class=\"hist-msg\">{snippet}</span></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.result  = None
        st.rerun()

st.markdown("<hr class=\"div\"><div class=\"footer\">SPAMSHIELD AI · MULTINOMIAL NAIVE BAYES · TF-IDF · BUILT WITH STREAMLIT</div>", unsafe_allow_html=True)
