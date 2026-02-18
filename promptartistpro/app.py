"""
app.py — Prompt Artist Pro (Streamlit UI)

A chatbot interface that classifies user intent using a trained
SVM model and retrieves optimized prompt templates from a local library.

Pure ML/NLP — No LLMs, No Generative AI APIs.

Usage:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import joblib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessor import preprocess
from prompt_dispatcher import PromptDispatcher

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'intent_classifier.joblib')
BOT_NAME = "Cipher"

CATEGORY_ICONS = {
    'CODING': '💻',
    'TESTING': '🧪',
    'FINANCE': '📊',
    'DATA_RECONCILIATION': '🔄',
    'MARKETING': '📣',
    'GENERAL': '🌐',
    'UI_UX_DESIGN': '🎨',
    'CYBERSECURITY': '🛡️',
    'DATA_SCIENCE': '🔬',
    'PROJECT_MANAGEMENT': '📅',
    'WRITING_CREATIVE': '✍️',
    'DATABASE': '🗄️',
}

CATEGORY_COLORS = {
    'CODING': '#61DAFB',
    'TESTING': '#98C379',
    'FINANCE': '#E5C07B',
    'DATA_RECONCILIATION': '#56B6C2',
    'MARKETING': '#E06C75',
    'GENERAL': '#ABB2BF',
    'UI_UX_DESIGN': '#FF79C6',
    'CYBERSECURITY': '#FF5555',
    'DATA_SCIENCE': '#BD93F9',
    'PROJECT_MANAGEMENT': '#F1FA8C',
    'WRITING_CREATIVE': '#8BE9FD',
    'DATABASE': '#FFB86C',
}

def generate_natural_response(intent: str, confidence: dict, templates: list, query: str) -> str:
    """
    Generates a context-aware, full-scale chatbot response.
    Analyzes intent, confidence, and the specific results found.
    """
    import random
    
    count = len(templates)
    query_lower = query.lower().strip()
    greetings = {'hi', 'hello', 'hey', 'greetings', 'sup', 'yo', 'morning', 'afternoon', 'evening'}
    
    # Check if it's a pure greeting
    is_greeting = any(g in query_lower for g in greetings) and len(query_lower.split()) <= 2

    # High-precision fallback for Greetings
    if is_greeting and intent == 'GENERAL':
        return f"Hello there! I'm **{BOT_NAME}**, your dedicated interaction architect. I've identified your greeting and prepared our core system introduction blueprints for you. How can I help you build something great today?"

    # Contextual acknowledgments
    if count == 0:
        return f"I've analyzed your query focusing on **{intent}**, but it seems I don't have a specific blueprint that matches those exact keywords. I can help you explore the library, or you can try a broader technical term."

    top_template_title = templates[0]['title'] if count > 0 else ""
    
    # Domain-specific conversational expansions
    responses = {
        'CODING': [
            f"I've mapped your request to **Software Engineering**. My top recommendation is the `{top_template_title}` blueprint, but I've found {count} relevant patterns in total. Which one shall we implement?",
            f"Technical analysis complete. To help with your {query_lower}, I've retrieved {count} structural blueprints. The `{top_template_title}` model might be the best starting point.",
        ],
        'CYBERSECURITY': [
            f"Security is my priority. I've identified intent related to **System Defense**. I've retrieved {count} hardened blueprints, including `{top_template_title}`. Let's secure your perimeter.",
            f"Scanning our security library... I've found {count} expert templates. Given your query, I highly recommend looking at `{top_template_title}` for robust protection.",
        ],
        'GENERAL': [
            f"I'm **{BOT_NAME}**. I've categorized this as a general inquiry. I've found {count} relevant resources for you, starting with `{top_template_title}`. How would you like to proceed?",
            f"Greetings! As your interaction architect, I've curated {count} general-purpose blueprints that match your request. I recommend starting with `{top_template_title}`.",
        ]
    }
    
    # Default fallback for other categories
    default_responses = [
        f"I've analyzed your request for **{intent}** and found {count} expert blueprints. The `{top_template_title}` template appears to be the most relevant to your goals.",
        f"Mapping to **{intent}**... I've prepared {count} specialized blueprints for you. Would you like to dive into the `{top_template_title}` methodology?",
        f"Context analysis complete. I'm confident identifying this as a **{intent}** task. Here are {count} blueprints to guide your workflow."
    ]
    
    domain_responses = responses.get(intent, default_responses)
    return random.choice(domain_responses)

# ── Model Loading (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained intent classifier pipeline."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    artifact = joblib.load(MODEL_PATH)
    return artifact['pipeline'], artifact['label_encoder']


# ── System Drivers ────────────────────────────────────────────────────────
# VERSION: Incremented to force st.cache_resource invalidation for new search logic
ENGINE_VERSION = "2.1.0-RegexFix"

@st.cache_resource
def load_dispatcher(version: str):
    """Load the prompt template dispatcher."""
    return PromptDispatcher()


# ── Classification Engine ─────────────────────────────────────────────────
def classify_intent(text: str, pipeline, label_encoder) -> tuple:
    """
    Classify user intent and return (category, confidence_scores).

    Returns:
        (predicted_category, confidence_dict)
    """
    processed = preprocess(text)
    if not processed.strip():
        return 'GENERAL', {'GENERAL': 1.0}

    # Get decision function scores
    decision_scores = pipeline.decision_function([processed])[0]

    # Convert to pseudo-probabilities via softmax
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()

    # Map to class names
    classes = label_encoder.classes_
    confidence = {cls: float(prob) for cls, prob in zip(classes, probabilities)}

    # Predicted class
    predicted_idx = np.argmax(decision_scores)
    predicted = classes[predicted_idx]

    return predicted, confidence


# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Artist Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;600;800&family=JetBrains+Mono&display=swap');

    /* ── Global Architecture ── */
    .stApp {
        font-family: 'Outfit', sans-serif;
        background: #0f172a;
        color: #ffffff !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0, transparent 50%), 
            radial-gradient(at 50% 0%, rgba(192, 132, 252, 0.1) 0, transparent 50%),
            radial-gradient(at 100% 0%, rgba(244, 114, 182, 0.15) 0, transparent 50%);
    }

    /* ── System Header Stealth Mode ── */
    header[data-testid="stHeader"] {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    header[data-testid="stHeader"] * {
        color: #ffffff !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Header Rebrand ── */
    .engine-header {
        text-align: center;
        padding: 2.5rem 0 3.5rem 0;
        animation: fadeInDown 0.8s ease-out;
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .engine-header h1 {
        font-size: 3.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #FF79C6, #BD93F9, #8BE9FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
        letter-spacing: -1.5px !important;
        text-shadow: 0 10px 30px rgba(189, 147, 249, 0.3) !important;
    }
    .engine-subtitle {
        color: #ABB2BF !important;
        font-size: 1.2rem !important;
        font-weight: 300 !important;
        letter-spacing: 0.5px !important;
        max-width: 800px;
        margin: 0 auto !important;
    }

    /* ── Template Card (Glassmorphism) ── */
    .template-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.8rem;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .template-card:hover {
        background: rgba(30, 41, 59, 0.65);
        border-color: rgba(139, 92, 246, 0.5);
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 25px rgba(139, 92, 246, 0.2);
    }
    .template-title {
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        letter-spacing: -0.5px;
    }
    .template-desc {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 400;
        line-height: 1.6;
        margin-bottom: 1.2rem;
        opacity: 0.9;
    }
    .template-body {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 12px;
        padding: 1.25rem;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        line-height: 1.7;
        border: 1px solid rgba(255, 255, 255, 0.05);
        max-height: 180px;
        overflow: hidden;
        position: relative;
    }
    .template-body::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 40px;
        background: linear-gradient(transparent, rgba(15, 23, 42, 0.9));
    }

    /* ── UI Elements Overhaul ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30, 41, 59, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 50px !important;
        padding: 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin: 0 auto 2.5rem !important;
        width: fit-content !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1 !important;
        border-radius: 50px !important;
        padding: 0.6rem 2.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }

    .stButton > button {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 14px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-2px);
    }

    .stTextInput input {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        padding: 0.8rem 1.2rem !important;
    }
    .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3) !important;
    }

    /* ── Chat & Results ── */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.5) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 1.5rem !important;
        padding: 1.5rem !important;
    }
    .stChatMessageAssistant {
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.1) !important;
    }

    .intent-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .explorer-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
        gap: 1.8rem;
        padding: 1.5rem 0;
    }

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.9) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    /* Force all text to white */
    p, li, h1, h2, h3, span, label, b, strong {
        color: #ffffff !important;
    }
    
    /* Code block consistency */
    .stCodeBlock, code, pre {
        background: rgba(15, 23, 42, 0.8) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* ── Selectbox Styling (Round 3 Fix) ── */
    /* Target the Streamlit selectbox container */
    .stSelectbox [data-baseweb="select"] {
        background-color: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 14px !important;
    }

    /* Force background for all nested divs to transparent to prevent white layers */
    .stSelectbox [data-baseweb="select"] div {
        background-color: transparent !important;
    }

    /* Target all text elements within the selectbox */
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div {
        color: #ffffff !important;
    }

    /* Target the dropdown arrow icon */
    .stSelectbox [data-baseweb="select"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Target the dropdown list (popover) */
    div[data-baseweb="popover"] ul {
        background-color: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="popover"] li {
        color: #ffffff !important;
    }

    div[data-baseweb="popover"] li:hover {
        background-color: rgba(99, 102, 241, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
        <div style="background:#818cf8; width:40px; height:40px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.5rem;">🎨</div>
        <div style="font-weight:700; font-size:1.1rem; color:#ffffff;">Engine Controls</div>
    </div>
    """, unsafe_allow_html=True)

    # Load models and dispatcher
    pipeline, label_encoder = load_model()
    # Pass version to force reload if engine logic changes
    dispatcher = load_dispatcher(ENGINE_VERSION)
    
    model_ready = pipeline is not None and label_encoder is not None

    if not model_ready:
        st.error("⚠️ No trained model found!\n\nRun `python train_model.py` first.")
        st.code("python train_model.py", language="bash")
        model_ready = False
    else:
        st.success("✅ Model loaded")
        model_ready = True

    st.markdown("---")
    st.markdown("### 📚 Available Domains")

    categories = dispatcher.get_categories()
    for cat in categories:
        info = dispatcher.get_category_info(cat)
        icon = CATEGORY_ICONS.get(cat, '📄')
        if info:
            st.markdown(f"**{icon} {cat}** ({info['template_count']})")

    st.markdown("---")
    st.markdown("### 🔧 Technical Stack")
    st.markdown("""
    - **Classifier**: SVM (LinearSVC)
    - **Features**: TF-IDF (1,2-grams)
    - **Preprocessing**: NLTK
    - **No LLMs** • No API calls
    """)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Rendering Utilities ────────────────────────────────────────────────────
def render_message(msg):
    """Render a single message with its classification results."""
    with st.chat_message(msg['role'], avatar="🎨" if msg['role'] == 'assistant' else "👤"):
        st.markdown(msg['content'])

        if msg.get('templates') and msg['intent']:
            # Show intent badge
            color = CATEGORY_COLORS.get(msg['intent'], '#ABB2BF')
            icon = CATEGORY_ICONS.get(msg['intent'], '📄')
            st.markdown(
                f"<span class='intent-badge' style='background: {color}22; "
                f"color: {color}; border: 1px solid {color}55;'>"
                f"{icon} {msg['intent']}</span>",
                unsafe_allow_html=True
            )

            # Confidence scores
            if msg.get('confidence'):
                render_confidence(msg['confidence'], msg['intent'])

            # Templates
            if msg['templates']:
                for i, tmpl in enumerate(msg['templates']):
                    # Create a unique key per message and template index
                    msg_id = hash(msg['content']) % 10000
                    render_template_card(tmpl, context_key=f"msg_{msg_id}_{i}")


def render_confidence(confidence: dict, predicted: str):
    """Render confidence score bars."""
    # Sort by score descending, show top 4
    sorted_conf = sorted(confidence.items(), key=lambda x: x[1], reverse=True)[:4]

    html = '<div style="background: rgba(30, 41, 59, 0.3); border-radius: 12px; padding: 1rem; border: 1px solid rgba(255,255,255,0.05); margin: 0.5rem 0;">'
    html += '<div style="color:#ffffff;font-size:0.75rem;margin-bottom:0.8rem;font-weight:700;letter-spacing:1px;">CONFIDENCE SCORES</div>'
    for cls, score in sorted_conf:
        color = CATEGORY_COLORS.get(cls, '#ABB2BF')
        icon = CATEGORY_ICONS.get(cls, '📄')
        width_pct = max(score * 100, 2)
        is_predicted = "★ " if cls == predicted else ""
        html += f'''
        <div style="display: flex; align-items: center; margin: 0.4rem 0; font-size: 0.8rem;">
            <span style="width: 160px; color: #ffffff; font-weight: 500;">{is_predicted}{icon} {cls}</span>
            <div style="flex: 1; height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; margin: 0 0.8rem;">
                <div style="height: 100%; width:{width_pct}%; background:{color}; border-radius: 3px;"></div>
            </div>
            <span style="width: 40px; text-align: right; color: #ffffff; font-weight: 600;">{score:.0%}</span>
        </div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_template_card(template: dict, context_key: str = "card"):
    """Render a single prompt template card."""
    # Create a stable ID for this template instance
    card_id = f"{context_key}_{hash(template['template']) % 100000}"
    
    with st.container():
        # Check if this card is currently selected for Focus Mode
        if st.session_state.get('selected_template_id') == card_id:
            render_focus_mode(template, f"focus_{card_id}")
        else:
            st.markdown(f"""
            <div class="template-card">
                <div class="template-title">📝 {template['title']}</div>
                <div class="template-desc">{template.get('description', '')}</div>
                <div class="template-body">{template['template'][:250]}...</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Customize & Copy", key=f"btn_{card_id}", use_container_width=True):
                st.session_state.selected_template_id = card_id
                st.session_state.selected_template = template # Keep for backwards compatibility if needed
                st.rerun()


# ── Main Content ───────────────────────────────────────────────────────────
st.markdown("""
<div class="engine-header">
    <h1>🎨 Prompt Artist Pro</h1>
    <p class="engine-subtitle">The ultimate blueprint engine for high-precision interaction architecture.</p>
</div>
""", unsafe_allow_html=True)

# ── Focus Mode Helper ──────────────────────────────────────────────────────
def render_focus_mode(t: dict, key_prefix: str):
    """Render the customization form for a template."""
    with st.container():
        st.markdown(f"""
        <div class="focus-mode-header">
            <h3 style="margin:0; color:#ffffff;">⚡ Focus Mode: {t['title']}</h3>
            <p style="margin:0.5rem 0 0 0; color:#cbd5e1; font-size:0.9rem;">{t.get('description', 'Customizing expert blueprint')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        import re
        placeholders = list(set(re.findall(r'\{(.*?)\}', t['template'])))
        
        col1, col2 = st.columns([1, 1])
        
        form_data = {}
        with col1:
            st.markdown("**1. Configure Parameters**")
            for p in placeholders:
                label = p.replace('_', ' ').title()
                form_data[p] = st.text_input(f"{label}", placeholder=f"Enter {label.lower()}...", key=f"{key_prefix}_{p}")
        
        with col2:
            st.markdown('<div class="preview-header">2. Final Blueprint</div>', unsafe_allow_html=True)
            
            # Construct final prompt
            actual_final_prompt = t['template']
            for p, val in form_data.items():
                if val:
                    actual_final_prompt = actual_final_prompt.replace('{'+p+'}', val)
            
            st.code(actual_final_prompt, language="markdown")
            st.caption("☝️ Click the square icon in the top-right to copy.")
            
            if st.button("🏁 Finish & Close", use_container_width=True, type="primary", key=f"{key_prefix}_close"):
                st.session_state.selected_template_id = None
                st.session_state.selected_template = None
                st.toast("✨ Prompt ready!")
                st.rerun()
    st.markdown("---")

# ── Tabs Configuration ─────────────────────────────────────────────────────
tab_chat, tab_explorer = st.tabs(["💬 AI Interaction", "🧭 Template Explorer"])

with tab_chat:
    # ── Initialize Chat History ────────────────────────────────────────────────
    if 'messages' not in st.session_state or not st.session_state.messages:
        st.session_state.messages = []

        # Add welcome message
        welcome_templates = dispatcher.get_templates('GENERAL')
        welcome_text = ""
        for t in welcome_templates:
            if t['title'] == 'Welcome Message':
                welcome_text = t['template']
                break

        if not welcome_text:
            welcome_text = (
                f"Greetings! I am **{BOT_NAME}**, your dedicated interaction architect. 🚀\n\n"
                "I specialize in crafting high-end AI blueprints tailored to your specific technical and creative needs. "
                "Describe your objective, and I will instantly analyze your intent to provide optimized, domain-expert templates."
            )

        st.session_state.messages.append({
            'role': 'assistant',
            'content': welcome_text,
            'intent': None,
            'confidence': None,
            'templates': None,
        })

    # Render Chat
    for msg in st.session_state.messages:
        render_message(msg)

    # Chat Input
    if user_input := st.chat_input("What are you working on? (e.g. 'Security audit for a cloud app')"):
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'intent': None,
            'confidence': None,
            'templates': None,
        })

        if model_ready:
            intent, confidence = classify_intent(user_input, pipeline, label_encoder)
            # Pass user_input as query for hybrid ranking
            all_templates = dispatcher.get_templates(intent, query=user_input)
            
            # Limit chat templates to Top 5 for better UX
            templates = all_templates[:5]

            top_confidence = max(confidence.values())
            icon = CATEGORY_ICONS.get(intent, '📄')

            # Generate natural language response
            response_text = generate_natural_response(intent, confidence, templates, user_input)

            st.session_state.messages.append({
                'role': 'assistant',
                'content': response_text,
                'intent': intent,
                'confidence': confidence,
                'templates': templates,
            })
        st.rerun()

with tab_explorer:
    st.markdown("### 🧭 Library Explorer")
    
    col_search, col_filter = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("🔍 Search templates...", placeholder="Search by title, role, or keyword...")
    with col_filter:
        cat_filter = st.selectbox("📁 Filter by Category", ["All Categories"] + list(CATEGORY_ICONS.keys()))

    # Get and rank templates
    if search_query:
        # Use our new ranked search
        search_results = dispatcher.search_templates(search_query)
        explorer_templates = []
        for cat, t in search_results:
            if cat_filter != "All Categories" and cat != cat_filter:
                continue
            t['category'] = cat
            explorer_templates.append(t)
    else:
        # Show all (filtered)
        all_categories = CATEGORY_ICONS.keys()
        explorer_templates = []
        for cat in all_categories:
            if cat_filter != "All Categories" and cat != cat_filter:
                continue
            tmpls = dispatcher.get_templates(cat)
            for t in tmpls:
                t['category'] = cat
                explorer_templates.append(t)

    st.markdown(f"**Found {len(explorer_templates)} templates** matching your search.")
    
    # Render templates in a grid
    st.markdown('<div class="explorer-grid">', unsafe_allow_html=True)
    for i, t in enumerate(explorer_templates):
        render_template_card(t, context_key=f"exp_{i}")
    st.markdown('</div>', unsafe_allow_html=True)

