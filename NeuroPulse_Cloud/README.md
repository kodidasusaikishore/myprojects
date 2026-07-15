# 🧠 NeuroPulse — 1-Minute Brain Gym

A rapid-fire cognitive training web app with **6 brain games**, live leaderboards, and an admin panel — built with **Streamlit** + a custom HTML/CSS/JS frontend component.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🎮 Games

| Game | Description |
|------|-------------|
| 🎨 **Color Match (Stroop)** | Identify the INK COLOR of a printed word, ignoring its meaning |
| 🔍 **Word Find (8-Way)** | Find hidden words across an 8-directional letter grid |
| 💡 **Memory Flash** | Memorize tile patterns and recall their positions |
| ➕ **Speed Calc** | Solve arithmetic equations with streak multipliers |
| 🔀 **Rule Switch** | Match shapes or colors based on shifting rules |
| ⚡ **Chrono Rush** | Tap numbers from lowest to highest in scattered tiles |

---

## 🚀 Deploy on Streamlit Community Cloud

1. **Fork or push this folder** to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point to your repo, set branch to `main`, and set the **Main file path** to `app.py`
4. Click **Deploy!**

> **Note:** Streamlit Cloud uses an ephemeral filesystem. The SQLite database stored in `/tmp` resets on each cold start (new deploy or after inactivity). Scores and config changes persist only within a single session lifecycle.

---

## 🏃 Run Locally

```bash
# Install dependency
pip install streamlit

# Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🗂️ Project Structure

```
Neuro_Pulse_Streamlit/
├── app.py               # Streamlit backend — DB logic, component declaration
├── schema.sql           # SQLite table definitions
├── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html       # Game UI shell
│   ├── css/
│   │   └── style.css    # Dark glassmorphism design system
│   └── js/
│       ├── audio.js     # Web Audio API synthesizer
│       ├── games.js     # All 6 game engines
│       └── app.js       # App orchestrator (postMessage ↔ Streamlit)
```

---

## ⚙️ Admin Panel

Click the **⚙️** icon in the header. Default passcode: **`1234`**

Admin features:
- Adjust game timer (30–180 seconds)
- Toggle adaptive difficulty & cheat mode
- Configure Speed Calc operators and Word Find grid size
- Change admin passcode
- Reset all leaderboard records

---

## 🛠️ Tech Stack

- **Backend:** Python · Streamlit · SQLite3
- **Frontend:** Vanilla HTML · CSS (glassmorphism) · JavaScript
- **Audio:** Web Audio API (no external files)
- **Communication:** Streamlit Custom Component (`postMessage` bidirectional RPC)
