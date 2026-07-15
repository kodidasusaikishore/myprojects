import streamlit as st
import os
import re
import streamlit.components.v1 as components

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroPulse - 1-Minute Brain Gym",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Strip ALL Streamlit chrome — the game is fullscreen
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"] > .main {
    padding: 0 !important;
}
[data-testid="stAppViewBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Build self-contained HTML (cached so it only reads files once) ───────────
@st.cache_data
def build_game_html():
    base = os.path.dirname(__file__)

    css      = open(os.path.join(base, "frontend", "css", "style.css"),  encoding="utf-8").read()
    audio_js = open(os.path.join(base, "frontend", "js", "audio.js"),    encoding="utf-8").read()
    games_js = open(os.path.join(base, "frontend", "js", "games.js"),    encoding="utf-8").read()
    app_js   = open(os.path.join(base, "frontend", "js", "app.js"),      encoding="utf-8").read()

    # Extract just the <body> content from index.html
    raw  = open(os.path.join(base, "frontend", "index.html"), encoding="utf-8").read()
    m    = re.search(r"<body>(.*?)</body>", raw, re.DOTALL)
    body = m.group(1).strip() if m else raw
    # Remove old external <script> tags (we inline everything)
    body = re.sub(r'<script[^>]*src="[^"]*"[^>]*></script>', "", body)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeuroPulse - 1-Minute Brain Gym</title>
  <meta name="description" content="NeuroPulse: Boost your logic, memory, focus, and arithmetic skills in rapid-fire 1-minute daily brain exercises.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
  <style>{css}</style>
</head>
<body>
{body}
<script>
/* ============================================================
   NeuroPulse Audio Engine
   ============================================================ */
{audio_js}
</script>
<script>
/* ============================================================
   NeuroPulse Game Engines (Color, Word, Memory, Math, Rule, Chrono)
   ============================================================ */
{games_js}
</script>
<script>
/* ============================================================
   NeuroPulse App Orchestrator
   ============================================================ */
{app_js}
</script>
</body>
</html>"""

# ─── Render ───────────────────────────────────────────────────────────────────
components.html(build_game_html(), height=900, scrolling=True)
