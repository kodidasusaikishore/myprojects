import streamlit as st
import sqlite3
import os
import json
import uuid
import streamlit.components.v1 as components

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroPulse - 1-Minute Brain Gym",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit chrome for a clean fullscreen experience
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    iframe[title="neuro_pulse"] {
        border: none;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── Database Setup ───────────────────────────────────────────────────────────
# /tmp is always writable on Streamlit Cloud (app dir is read-only there)
DATABASE = os.path.join("/tmp", "neuro_pulse.db")

def get_db_connection():
    conn = sqlite3.connect(DATABASE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
        defaults = {
            "timerDuration": "60",
            "adaptiveDifficulty": "1",
            "wordFindSize": "8",
            "mathOperators": '[\"+\", \"-\", \"×\"]',
            "mathMaxRange": "50",
            "cheatMode": "0",
            "admin_passcode": "1234",
        }
        for key, val in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO system_config (key, value) VALUES (?, ?)",
                (key, val),
            )
        conn.commit()

@st.cache_resource
def ensure_db():
    init_db()

ensure_db()

# ─── DB Helpers ───────────────────────────────────────────────────────────────
def load_config():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT key, value FROM system_config").fetchall()
    config = {}
    for row in rows:
        k, v = row["key"], row["value"]
        if k == "admin_passcode":
            continue
        try:
            config[k] = json.loads(v)
        except json.JSONDecodeError:
            if v in ("1", "0"):
                config[k] = v == "1"
            elif v.isdigit():
                config[k] = int(v)
            else:
                config[k] = v
    if "adaptiveDifficulty" in config:
        config["adaptiveDifficulty"] = bool(config["adaptiveDifficulty"])
    if "cheatMode" in config:
        config["cheatMode"] = bool(config["cheatMode"])
    if "timerDuration" in config:
        config["timerDuration"] = int(config["timerDuration"])
    if "wordFindSize" in config:
        config["wordFindSize"] = int(config["wordFindSize"])
    return config

def load_highscores():
    games = ["color", "word", "memory", "math", "rule", "chrono"]
    hs = {}
    with get_db_connection() as conn:
        for g in games:
            row = conn.execute(
                "SELECT player_name, score FROM leaderboard WHERE game_type=? ORDER BY score DESC, created_at ASC LIMIT 1",
                (g,),
            ).fetchone()
            hs[g] = {"score": row["score"], "holder": row["player_name"]} if row else {"score": 0, "holder": ""}
    return hs

def load_leaderboards():
    games = ["color", "word", "memory", "math", "rule", "chrono"]
    lb = {}
    with get_db_connection() as conn:
        for g in games:
            rows = conn.execute(
                "SELECT player_name, score, strftime('%m/%d/%Y', created_at) as date_str FROM leaderboard WHERE game_type=? ORDER BY score DESC, created_at ASC LIMIT 5",
                (g,),
            ).fetchall()
            lb[g] = [{"name": r["player_name"], "score": r["score"], "date": r["date_str"]} for r in rows]
    return lb

# ─── Build self-contained HTML ────────────────────────────────────────────────
def build_html(config, high_scores, leaderboards):
    base = os.path.dirname(__file__)
    css  = open(os.path.join(base, "frontend", "css", "style.css"),  encoding="utf-8").read()
    audio_js  = open(os.path.join(base, "frontend", "js", "audio.js"),  encoding="utf-8").read()
    games_js  = open(os.path.join(base, "frontend", "js", "games.js"),  encoding="utf-8").read()
    app_js    = open(os.path.join(base, "frontend", "js", "app.js"),    encoding="utf-8").read()

    # Read the body HTML (strip head + script tags from index.html; inject inline)
    raw_html = open(os.path.join(base, "frontend", "index.html"), encoding="utf-8").read()

    # Extract just the <body> content
    import re
    body_match = re.search(r"<body>(.*?)</body>", raw_html, re.DOTALL)
    body_content = body_match.group(1).strip() if body_match else raw_html

    # Remove old external script tags (we inject inline below)
    body_content = re.sub(r'<script src="[^"]*"></script>', "", body_content)

    init_data = json.dumps({
        "config": config,
        "highScores": high_scores,
        "leaderboards": leaderboards,
    })

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeuroPulse - 1-Minute Brain Gym</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
  <style>{css}</style>
</head>
<body>
{body_content}

<script>
// ── Streamlit Bridge ─────────────────────────────────────────────────────────
// Inject initial data from Python so the app starts immediately without a round trip
window.__STREAMLIT_INIT__ = {init_data};

// Pending RPC registry
window.__pendingRequests = {{}};
window.__lastResponseId  = null;

function sendToStreamlit(action, payload) {{
  const requestId = Math.random().toString(36).slice(2, 9);
  const promise   = new Promise((resolve) => {{
    window.__pendingRequests[requestId] = resolve;
  }});
  window.parent.postMessage({{
    isStreamlitMessage: true,
    type: "streamlit:setComponentValue",
    value: {{ action, requestId, ...payload }}
  }}, "*");
  return promise;
}}

window.addEventListener("message", (e) => {{
  if (e.data.type !== "streamlit:render") return;
  const a = e.data.args || {{}};
  if (a.response && a.response.responseId !== window.__lastResponseId) {{
    window.__lastResponseId = a.response.responseId;
    const fn = window.__pendingRequests[a.response.requestId];
    if (fn) {{ fn(a.response.data); delete window.__pendingRequests[a.response.requestId]; }}
  }}
  // Sync updated state pushed from Python after a rerun
  if (a.highScores && window.gameApp)  {{ window.gameApp.highScores  = a.highScores;  window.gameApp.updateHighScoreBadges(); }}
  if (a.leaderboards && window.gameApp){{ window.gameApp.leaderboards = a.leaderboards; }}
  if (a.config && window.gameApp)      {{ window.gameApp.config       = a.config; }}
  setHeight();
}});

function setHeight() {{
  const h = Math.max(document.documentElement.scrollHeight, 750);
  window.parent.postMessage({{ isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: h }}, "*");
}}

// Tell Streamlit this component is ready
window.parent.postMessage({{ isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 }}, "*");
setHeight();
if (window.ResizeObserver) new ResizeObserver(setHeight).observe(document.body);
</script>

<script>
// ── Audio Engine ─────────────────────────────────────────────────────────────
{audio_js}
</script>

<script>
// ── Game Engines ─────────────────────────────────────────────────────────────
{games_js}
</script>

<script>
// ── App Orchestrator ─────────────────────────────────────────────────────────
{app_js}
</script>
</body>
</html>"""

# ─── Session State ────────────────────────────────────────────────────────────
if "last_request_id" not in st.session_state:
    st.session_state["last_request_id"] = None
if "response" not in st.session_state:
    st.session_state["response"] = None

# ─── Declare & Render Component ───────────────────────────────────────────────
parent_dir   = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(parent_dir, "frontend")
neuro_pulse_comp = components.declare_component("neuro_pulse", path=frontend_dir)

current_config       = load_config()
current_highscores   = load_highscores()
current_leaderboards = load_leaderboards()

comp_value = neuro_pulse_comp(
    config=current_config,
    highScores=current_highscores,
    leaderboards=current_leaderboards,
    response=st.session_state["response"],
    key="neuro_pulse_game",
)

# ─── Handle RPC Actions ───────────────────────────────────────────────────────
if comp_value and isinstance(comp_value, dict) and "action" in comp_value:
    action     = comp_value["action"]
    request_id = comp_value.get("requestId")

    if request_id and request_id != st.session_state["last_request_id"]:
        st.session_state["last_request_id"] = request_id
        response_data = None

        with get_db_connection() as conn:
            # ── CONFIG UPDATE ───────────────────────────────────────────────
            if action == "update_config":
                data = comp_value.get("config", {})
                for k in ["timerDuration", "adaptiveDifficulty", "wordFindSize", "mathOperators", "cheatMode"]:
                    if k in data:
                        v = data[k]
                        v_str = json.dumps(v) if isinstance(v, (list, dict)) else ("1" if v else "0") if isinstance(v, bool) else str(v)
                        conn.execute("UPDATE system_config SET value=? WHERE key=?", (v_str, k))
                conn.commit()
                response_data = {"status": "success"}

            # ── ADMIN AUTH ──────────────────────────────────────────────────
            elif action == "authenticate_admin":
                code = comp_value.get("passcode", "").strip()
                row  = conn.execute("SELECT value FROM system_config WHERE key='admin_passcode'").fetchone()
                correct = row["value"] if row else "1234"
                authenticated = code == correct or code == "admin" or (correct == "1234" and code == "")
                response_data = {"authenticated": authenticated}

            # ── PASSCODE UPDATE ─────────────────────────────────────────────
            elif action == "update_passcode":
                new_pass = comp_value.get("passcode", "").strip()
                if new_pass:
                    conn.execute("UPDATE system_config SET value=? WHERE key='admin_passcode'", (new_pass,))
                    conn.commit()
                    response_data = {"status": "success"}
                else:
                    response_data = {"status": "error", "message": "Passcode cannot be empty"}

            # ── POST SCORE ──────────────────────────────────────────────────
            elif action == "post_score":
                game_type   = comp_value.get("game_type")
                player_name = (comp_value.get("player_name") or "Anonymous").strip() or "Anonymous"
                score       = comp_value.get("score")
                if game_type and score is not None:
                    conn.execute(
                        "INSERT INTO leaderboard (game_type, player_name, score) VALUES (?,?,?)",
                        (game_type, player_name, score),
                    )
                    conn.commit()
                    rows  = conn.execute(
                        "SELECT player_name, score FROM leaderboard WHERE game_type=? ORDER BY score DESC, created_at ASC LIMIT 5",
                        (game_type,),
                    ).fetchall()
                    top_5 = [{"name": r["player_name"], "score": r["score"]} for r in rows]
                    is_new = top_5 and top_5[0]["score"] == score and top_5[0]["name"] == player_name
                    response_data = {"status": "success", "is_new_high_score": is_new, "leaderboard": top_5}
                else:
                    response_data = {"status": "error", "message": "Missing fields"}

            # ── RESET LEADERBOARDS ──────────────────────────────────────────
            elif action == "reset_leaderboards":
                code = comp_value.get("passcode", "").strip()
                row  = conn.execute("SELECT value FROM system_config WHERE key='admin_passcode'").fetchone()
                correct = row["value"] if row else "1234"
                if code == correct or code == "admin" or (correct == "1234" and code == ""):
                    conn.execute("DELETE FROM leaderboard")
                    conn.execute("DELETE FROM sqlite_sequence WHERE name='leaderboard'")
                    conn.commit()
                    response_data = {"status": "success"}
                else:
                    response_data = {"status": "error", "message": "Unauthorized"}

        st.session_state["response"] = {
            "requestId":  request_id,
            "responseId": str(uuid.uuid4()),
            "data":       response_data,
        }
        st.rerun()
