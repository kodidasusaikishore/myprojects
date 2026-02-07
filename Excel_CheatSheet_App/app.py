import streamlit as st
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="Excel Shortcuts Master", page_icon="⌨️", layout="wide")

# --- DATA: SHORTCUT DATABASE ---
@st.cache_data
def load_data():
    try:
        # Load from local JSON file
        import json
        import os
        file_path = os.path.join(os.path.dirname(__file__), 'shortcuts_data.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            shortcuts = json.load(f)
        return pd.DataFrame(shortcuts)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# --- CUSTOM CSS (NEON THEME) ---
st.markdown("""
<style>
    /* Main Background - Adjusted */
    .stApp {
        background-color: #0b1116; /* Slightly lighter than pure black */
        background-image: radial-gradient(circle at 50% 50%, #15221e 0%, #050a08 100%);
        color: #ffffff;
    }

    /* Input Fields - Hyper Nuclear Fix v2 */
    .stTextInput input, 
    .stSelectbox input, 
    .stNumberInput input,
    input[data-baseweb="input"] {
        background-color: #0a0a0a !important;
        color: #00ff87 !important;
        caret-color: #00ff87 !important;
        -webkit-text-fill-color: #00ff87 !important;
        border-color: #00ff87 !important;
        font-weight: bold !important;
        text-shadow: 0 0 5px rgba(0, 255, 135, 0.3); /* Slight glow to make it pop */
    }
    
    /* Input Wrapper Background */
    div[data-baseweb="base-input"] {
        background-color: #0a0a0a !important;
        border: 1px solid #00ff87 !important;
        border-radius: 8px !important;
    }

    /* Placeholder Text */
    ::placeholder {
        color: rgba(0, 255, 135, 0.5) !important;
        -webkit-text-fill-color: rgba(0, 255, 135, 0.5) !important;
    }

    .stTextInput div[data-testid="stTextInput-root-element"] {
        background-color: #0a0a0a !important;
    }
    
    /* Dropdown - Fully Neon - Nuclear */
    div[data-baseweb="select"] > div {
        background-color: #0a0a0a !important;
        border: 1px solid #00ff87 !important;
        color: white !important;
    }
    div[data-baseweb="select"] span {
        color: #00ff87 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #00ff87 !important;
    }
    /* Dropdown Popover & Options - Final Attempt */
    /* Target the specific portal root if possible or use broad wildcard */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
    }
    
    /* Text inside options - wildcards */
    ul[role="listbox"] li * {
        color: #ffffff !important;
    }
    
    /* Highlight/Hover state - Fixed Text Contrast */
    li[role="option"]:hover, 
    li[role="option"][aria-selected="true"],
    li[role="option"]:hover *,
    li[role="option"][aria-selected="true"] * {
        background-color: #00ff87 !important;
        color: #000000 !important; /* Force Black Text on Green Background */
        font-weight: bold !important;
    }
    
    /* Scrollbar Glow */
    ::-webkit-scrollbar {
        width: 10px;
        background: #050505;
    }
    ::-webkit-scrollbar-thumb {
        background: #00ff87;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 255, 135, 0.5);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00cc6a;
    }
    
    /* Title Glow */
    h1 {
        text-align: center;
        font-weight: 800;
        background: linear-gradient(to right, #00ff87, #60efff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 255, 135, 0.5);
    }
    
    /* Neon Key Card */
    .key-card {
        background: rgba(20, 20, 20, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 135, 0.2);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 0 15px rgba(0, 255, 135, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .key-card:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: #00ff87;
        box-shadow: 0 0 30px rgba(0, 255, 135, 0.4);
    }
    
    .key-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(45deg, transparent, rgba(0,255,135,0.1), transparent);
        transform: translateX(-100%);
        transition: 0.5s;
    }
    
    .key-card:hover::before {
        transform: translateX(100%);
    }

    /* Shortcut Text */
    .shortcut-text {
        color: #00ff87; /* Neon Green */
        font-weight: 900;
        font-size: 1.4rem;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 10px rgba(0, 255, 135, 0.8);
        letter-spacing: 1px;
    }
    
    /* Action Text */
    .action-text {
        color: #e0e0e0;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 8px;
    }
    
    /* Category Tag */
    .category-tag {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: linear-gradient(90deg, #1d976c, #93f9b9);
        padding: 4px 10px;
        border-radius: 20px;
        color: #000;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 8px;
        box-shadow: 0 0 10px rgba(29, 151, 108, 0.4);
    }

    /* Keyboard Styles */
    .keyboard-container {
        display: flex;
        flex-direction: column;
        gap: 5px;
        background: #111;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #00ff87;
        box-shadow: 0 0 20px rgba(0, 255, 135, 0.1);
        margin-top: 20px;
    }
    .keyboard-row {
        display: flex;
        justify-content: center;
        gap: 5px;
    }
    .key {
        background: #222;
        border: 1px solid #444;
        border-radius: 6px;
        color: #ddd;
        font-family: monospace;
        font-weight: bold;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s;
        min-width: 40px;
        height: 40px;
        padding: 5px;
        user-select: none;
    }
    .key:hover {
        border-color: #00ff87;
        color: #00ff87;
        box-shadow: 0 0 10px rgba(0, 255, 135, 0.3);
        transform: translateY(-2px);
    }
    .key.active {
        background: #00ff87;
        color: black;
        box-shadow: 0 0 15px #00ff87;
        border-color: #00ff87;
    }
    .key-wide { flex-grow: 1.5; }
    .key-extra-wide { flex-grow: 2; }
    .key-space { flex-grow: 6; }
    /* Help Text (Press Enter to apply) */
    div[data-testid="stInputHelp"] {
        color: #00ff87 !important;
        font-family: 'Courier New', monospace;
    }
    .stTextInput div[data-baseweb="input"] + div {
        color: #00ff87 !important;
        opacity: 0.8;
    }
    
    /* Fix Button Text Visibility - Force Black Background */
    .stButton>button {
        background-color: #0a0a0a !important; 
        color: #00ff87 !important;
        border: 2px solid #00ff87 !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
    }
    .stButton>button:hover {
        background-color: #00ff87 !important;
        color: #000000 !important;
        border-color: #00ff87 !important;
        box-shadow: 0 0 15px rgba(0, 255, 135, 0.6);
        transform: scale(1.02);
    }
    
    /* Sidebar Styling - Force Deep Forest Match */
    section[data-testid="stSidebar"] {
        background-color: #101510 !important; /* Force override */
        border-right: 1px solid rgba(0, 255, 135, 0.2) !important;
    }
    
    /* Sidebar Text */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: #e0e0e0 !important;
    }
    
    /* Header Elements (Deploy, Stop, Menu, Running Man) - Nuclear White */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    header[data-testid="stHeader"] * {
        color: white !important;
    }
    header[data-testid="stHeader"] svg {
        fill: white !important;
        stroke: white !important;
    }
    
    /* Running Animation Status */
    div[data-testid="stStatusWidget"] {
        color: white !important;
    }
    div[data-testid="stStatusWidget"] svg {
        fill: white !important;
    }
    
    /* Sidebar Toggle Arrow - Nuclear White */
    button[kind="header"] {
        color: white !important;
    }
    button[kind="header"] svg {
        fill: white !important;
        stroke: white !important;
        color: white !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarExpandedControl"] svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }

    /* Scrollbar Glow - Chrome/Safari/Edge */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
        background: #050505;
    }
    
    /* Thumb - Always Green */
    ::-webkit-scrollbar-thumb {
        background-color: #00ff87 !important;
        background: #00ff87 !important; /* Duplicate for robustness */
        border-radius: 6px;
        border: 2px solid #050505;
        box-shadow: 0 0 10px rgba(0, 255, 135, 0.5);
    }
    
    /* Window Scrollbar specific target */
    body::-webkit-scrollbar-thumb {
        background-color: #00ff87 !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: #00ff87 !important;
        box-shadow: 0 0 15px rgba(0, 255, 135, 0.8);
    }
    ::-webkit-scrollbar-track {
        background: #111;
    }
    
    /* Firefox */
    * {
        scrollbar-width: thin;
        scrollbar-color: #00ff87 #050505;
    }
    
    /* Top Header Bar Transparent */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    .stTextInput label, 
    .stSelectbox label, 
    .stRadio label,
    .stRadio div[role="radiogroup"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-shadow: 0 0 5px rgba(255,255,255,0.2);
    }
    
    /* Expander Fix - Nuclear */
    div[data-testid="stExpander"] {
        background-color: #0a0a0a !important;
        border: 1px solid #00ff87 !important;
        border-radius: 8px !important;
        color: white !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: #ffffff !important;
        background-color: #0a0a0a !important;
    }
    .streamlit-expanderHeader {
        background-color: #0a0a0a !important;
        color: #00ff87 !important;
    }
    .streamlit-expanderHeader:hover {
        color: #ffffff !important;
        background-color: #0a0a0a !important;
    }
    .streamlit-expanderContent {
        background-color: #0a0a0a !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home (Search)", "🎮 Shortcut Dojo (Quiz)", "🃏 Flashcards"], label_visibility="collapsed")
st.sidebar.divider()

# --- GLOBAL PLATFORM STATE ---
# Initialize platform in session state if not exists
if "platform" not in st.session_state:
    st.session_state.platform = "Windows"

# Add Platform Switcher to Sidebar (Global)
st.sidebar.subheader("💻 Platform")
st.session_state.platform = st.sidebar.radio("Select OS:", ["Windows", "Mac"], index=0 if st.session_state.platform == "Windows" else 1)

platform = st.session_state.platform

# --- PAGE: HOME ---
if page == "🏠 Home (Search)":
    # --- HEADER ---
    st.title("⌨️ Excel Shortcut Master")
    st.markdown("The ultimate interactive cheat sheet for Excel power users.")

    col_search, col_filter = st.columns([3, 2])

    with col_search:
        search_query = st.text_input("🔍 Search (e.g. 'Paste', 'Filter', 'Row')", "")

    with col_filter:
        if not df.empty and 'Category' in df.columns:
            categories = ["All"] + sorted(df['Category'].unique().tolist())
        else:
            categories = ["All"]
        selected_category = st.selectbox("📂 Filter by Category", categories)

    # --- FILTER LOGIC ---
    filtered_df = df.copy()

    # 1. Search Filter
    if search_query:
        filtered_df = filtered_df[
            filtered_df['Action'].str.contains(search_query, case=False) | 
            filtered_df['Keys'].str.contains(search_query, case=False)
        ]

    # 2. Category Filter
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]

    # --- DISPLAY CARDS ---
    st.divider()

    if filtered_df.empty:
        st.warning("No shortcuts found matching your criteria.")
    else:
        # Responsive Grid
        cols = st.columns(3)
        
        for i, (index, row) in enumerate(filtered_df.iterrows()):
            col = cols[i % 3]
            
            # Select Key based on Platform
            key_display = row['Keys'] if platform == "Windows" else row['Mac']
            
            with col:
                st.markdown(f"""
                <div class="key-card">
                    <div class="category-tag">{row['Category']}</div>
                    <div class="shortcut-text">{key_display}</div>
                    <div class="action-text">{row['Action']}</div>
                </div>
                """, unsafe_allow_html=True)

# --- PAGE: VISUAL KEYBOARD --- Removed by request

# --- PAGE: QUIZ DOJO ---
elif page == "🎮 Shortcut Dojo (Quiz)":
    import random
    st.title("🎮 Shortcut Dojo")
    st.markdown("### Challenge: 5 Questions to prove your skills!")
    
    # Initialize Quiz Session
    if "quiz_active" not in st.session_state:
        st.session_state.quiz_active = False
        st.session_state.current_q_index = 0
        st.session_state.quiz_score = 0
        st.session_state.questions = []
        st.session_state.selected_option = None
        st.session_state.answer_submitted = False
    
    # Start/Restart Button
    if not st.session_state.quiz_active:
        if st.button("Start New Quiz 🚀"):
            st.session_state.quiz_active = True
            st.session_state.current_q_index = 0
            st.session_state.quiz_score = 0
            st.session_state.answer_submitted = False
            st.session_state.selected_option = None
            
            # Generate 5 unique questions
            questions = []
            for _ in range(5):
                q = df.sample(1).iloc[0]
                correct_key = q['Keys'] if platform == "Windows" else q['Mac']
                
                # Generate Distractors (Unique)
                distractors = set()
                attempts = 0
                while len(distractors) < 3 and attempts < 20:
                    d = df.sample(1).iloc[0]
                    d_key = d['Keys'] if platform == "Windows" else d['Mac']
                    if d_key != correct_key and d_key != "":
                        distractors.add(d_key)
                    attempts += 1
                
                options = [correct_key] + list(distractors)
                random.shuffle(options)
                
                questions.append({
                    "q": q['Action'],
                    "a": correct_key,
                    "cat": q['Category'],
                    "opts": options
                })
            st.session_state.questions = questions
            st.rerun()

    # Active Quiz UI
    else:
        # Check if finished
        if st.session_state.current_q_index >= 5:
            st.balloons()
            score = st.session_state.quiz_score
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: rgba(0,255,135,0.1); border: 2px solid #00ff87; border-radius: 10px;">
                <h1>🎉 Quiz Complete!</h1>
                <h2>Your Score: <span style="color: #00ff87;">{score}/50</span></h2>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Play Again 🔄"):
                st.session_state.quiz_active = False
                st.rerun()
        else:
            # Display Question
            q_idx = st.session_state.current_q_index
            q_data = st.session_state.questions[q_idx]
            
            st.progress((q_idx + 1) / 5, text=f"Question {q_idx + 1} of 5")
            
            st.markdown(f"### **{q_data['q']}**")
            st.caption(f"Category: {q_data['cat']}")
            
            # Radio Button (Reset state on new question)
            choice = st.radio("Choose the correct shortcut:", q_data['opts'], key=f"q_{q_idx}", disabled=st.session_state.answer_submitted)
            
            col_sub, col_next = st.columns([1, 1])
            
            with col_sub:
                if not st.session_state.answer_submitted:
                    if st.button("Submit Answer ✅"):
                        st.session_state.answer_submitted = True
                        if choice == q_data['a']:
                            st.success(f"Correct! 🎉")
                            st.session_state.quiz_score += 10
                        else:
                            st.error(f"Wrong! The correct answer was **{q_data['a']}**")
                        st.rerun()
                else:
                    # Show feedback persistently after submit
                    if choice == q_data['a']:
                         st.success(f"Correct! 🎉 (+10 pts)")
                    else:
                         st.error(f"Wrong! The correct answer was **{q_data['a']}**")

            with col_next:
                if st.session_state.answer_submitted:
                    if st.button("Next Question ⏭️"):
                        st.session_state.current_q_index += 1
                        st.session_state.answer_submitted = False
                        st.rerun()


# --- PAGE: FLASHCARDS ---
elif page == "🃏 Flashcards":
    st.title("🃏 Shortcut Flashcards")
    st.markdown("Flip the card to reveal the shortcut!")
    
    if "flashcard" not in st.session_state:
        st.session_state.flashcard = df.sample(1).iloc[0]
        st.session_state.flipped = False
    
    card = st.session_state.flashcard
    answer = card['Keys'] if platform == "Windows" else card['Mac']
    
    # Custom CSS for Flip Card
    st.markdown("""
    <style>
        .flashcard-container {
            perspective: 1000px;
            width: 100%;
            height: 300px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
        }
        .flashcard {
            width: 500px;
            height: 280px;
            background: rgba(10, 10, 10, 0.8);
            border: 2px solid #00ff87;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            box-shadow: 0 0 30px rgba(0, 255, 135, 0.2);
            padding: 20px;
            transition: transform 0.6s;
            transform-style: preserve-3d;
        }
        .fc-content {
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .fc-sub {
            color: #888;
            font-size: 0.9rem;
            margin-top: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .fc-answer {
            color: #00ff87;
            font-size: 3rem;
            font-family: 'Courier New', monospace;
            text-shadow: 0 0 20px rgba(0, 255, 135, 0.8);
        }
    </style>
    """, unsafe_allow_html=True)
    
    col_card, col_btn = st.columns([3, 1])
    
    with col_card:
        if st.button("🔄 FLIP CARD", use_container_width=True):
            st.session_state.flipped = not st.session_state.flipped
        
        # Display Card Content based on State
        if not st.session_state.flipped:
            # FRONT
            st.markdown(f"""
            <div class="flashcard">
                <div class="fc-content">{card['Action']}</div>
                <div class="fc-sub">Category: {card['Category']}</div>
                <div style="margin-top: 30px; font-size: 0.8rem; color: #555;">(Tap Flip to Reveal)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # BACK
            st.markdown(f"""
            <div class="flashcard" style="border-color: #00c6ff; box-shadow: 0 0 30px rgba(0, 198, 255, 0.3);">
                <div class="fc-sub" style="color: #00c6ff;">SHORTCUT</div>
                <div class="fc-answer" style="color: #00c6ff; text-shadow: 0 0 20px rgba(0, 198, 255, 0.8);">{answer}</div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_btn:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if st.button("⏭️ Next Card", use_container_width=True):
            st.session_state.flashcard = df.sample(1).iloc[0]
            st.session_state.flipped = False
            st.rerun()

