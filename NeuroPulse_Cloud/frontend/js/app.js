/* ============================================================
   NEUROPULSE — MAIN APP ENGINE & STATE MANAGEMENT
   All persistence is via localStorage — works fully offline
   and on Streamlit Cloud with zero backend round-trips.
   ============================================================ */

// ── localStorage key constants ────────────────────────────────────────────────
const LS_CONFIG   = 'np_config';
const LS_PASSCODE = 'np_admin_passcode';
const LS_LB_KEY   = (game) => `np_lb_${game}`;  // per-game leaderboard array
const GAMES       = ['color', 'word', 'memory', 'math', 'rule', 'chrono'];

// ── Default config ────────────────────────────────────────────────────────────
const DEFAULT_CONFIG = {
    timerDuration:     60,
    adaptiveDifficulty: true,
    wordFindSize:      8,
    mathOperators:     ['+', '-', '×'],
    mathMaxRange:      50,
    cheatMode:         false
};

// ── Storage helpers ───────────────────────────────────────────────────────────
function lsGet(key, fallback = null) {
    try {
        const v = localStorage.getItem(key);
        return v !== null ? JSON.parse(v) : fallback;
    } catch { return fallback; }
}
function lsSet(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
}

class App {
    constructor() {
        this.currentGame    = null;
        this.score          = 0;
        this.timeLeft       = 60;
        this.timerInterval  = null;
        this.activeLeaderboardTab = 'color';

        // Load config and scores from localStorage
        this.config     = { ...DEFAULT_CONFIG, ...(lsGet(LS_CONFIG) || {}) };
        this.highScores = this._computeHighScores();

        this.init();
    }

    init() {
        this.updateHighScoreBadges();
        this.updateAudioToggleUI();
    }

    // ── Config ────────────────────────────────────────────────────────────────
    _saveConfig() {
        lsSet(LS_CONFIG, this.config);
    }

    // ── High Scores (computed from leaderboards) ──────────────────────────────
    _computeHighScores() {
        const hs = {};
        GAMES.forEach(g => {
            const lb = lsGet(LS_LB_KEY(g), []);
            if (lb.length > 0) {
                hs[g] = { score: lb[0].score, holder: lb[0].name };
            } else {
                hs[g] = { score: 0, holder: '' };
            }
        });
        return hs;
    }

    updateHighScoreBadges() {
        GAMES.forEach(game => {
            const el = document.getElementById(`hs-${game}`);
            if (el && this.highScores[game]) {
                const { score, holder } = this.highScores[game];
                el.innerText = holder ? `${score} (${holder})` : score;
            }
        });
    }

    // ── Audio ─────────────────────────────────────────────────────────────────
    toggleAudio() {
        audio.enabled = !audio.enabled;
        this.updateAudioToggleUI();
    }

    updateAudioToggleUI() {
        const btn = document.getElementById('audioToggle');
        if (btn) btn.innerText = audio.enabled ? '🔊' : '🔇';
    }

    // ── Navigation ────────────────────────────────────────────────────────────
    showHub() {
        this.stopTimer();
        document.getElementById('stageScreen').classList.remove('active');
        document.getElementById('hubScreen').classList.add('active');
        document.getElementById('gameOverModal').classList.remove('active');
        document.getElementById('adminModal').classList.remove('active');
        document.getElementById('leaderboardsModal').classList.remove('active');
        this.highScores = this._computeHighScores();
        this.updateHighScoreBadges();
        this.currentGame = null;
    }

    // ── Game Lifecycle ────────────────────────────────────────────────────────
    startGame(gameType) {
        audio.click();
        this.currentGame = gameType;
        this.score       = 0;
        this.timeLeft    = this.config.timerDuration;
        document.getElementById('currentScore').innerText = '0';

        document.getElementById('hubScreen').classList.remove('active');
        document.getElementById('stageScreen').classList.add('active');
        document.getElementById('gameOverModal').classList.remove('active');
        document.getElementById('adminModal').classList.remove('active');
        document.getElementById('leaderboardsModal').classList.remove('active');

        const arena = document.getElementById('gameArena');
        arena.innerHTML = '';

        if      (gameType === 'color')  this.gameLogic = new ColorGame(arena, this);
        else if (gameType === 'word')   this.gameLogic = new WordGame(arena, this);
        else if (gameType === 'memory') this.gameLogic = new MemoryGame(arena, this);
        else if (gameType === 'math')   this.gameLogic = new MathGame(arena, this);
        else if (gameType === 'rule')   this.gameLogic = new RuleGame(arena, this);
        else if (gameType === 'chrono') this.gameLogic = new ChronoGame(arena, this);

        this.gameLogic.init();
        this.startTimer();
    }

    restartCurrentGame() {
        if (this.currentGame) this.startGame(this.currentGame);
    }

    addScore(pts) {
        this.score += pts;
        document.getElementById('currentScore').innerText = this.score;
    }

    // ── Timer ─────────────────────────────────────────────────────────────────
    startTimer() {
        this.stopTimer();
        this.updateTimerUI();
        this.timerInterval = setInterval(() => {
            this.timeLeft--;
            this.updateTimerUI();
            if (this.timeLeft <= 5 && this.timeLeft > 0) audio.tick();
            if (this.timeLeft <= 0) this.endGame(true);
        }, 1000);
    }

    stopTimer() {
        if (this.timerInterval) clearInterval(this.timerInterval);
    }

    updateTimerUI() {
        document.getElementById('timerText').innerText = this.timeLeft;
        const circle       = document.getElementById('timerProgress');
        const circumference = 2 * Math.PI * 35;
        const offset       = circumference - (this.timeLeft / this.config.timerDuration) * circumference;
        circle.style.strokeDashoffset = offset;
        circle.style.stroke = this.timeLeft <= 10 ? '#ef4444' : 'var(--accent-cyan)';
    }

    // ── Game Over ─────────────────────────────────────────────────────────────
    async endGame(timeIsUp = true) {
        this.stopTimer();
        audio.levelUp();

        // Read current leaderboard from localStorage
        const lb = lsGet(LS_LB_KEY(this.currentGame), []);
        const bestScore = lb.length > 0 ? lb[0].score : 0;
        const bestName  = lb.length > 0 ? lb[0].name  : '';

        const isNewPersonalBest      = lb.length === 0 || this.score > bestScore;
        const qualifiesForLeaderboard = lb.length < 5 || this.score > (lb[lb.length - 1]?.score || 0);

        // Modal header
        if (timeIsUp) {
            document.getElementById('modalIcon').innerText     = '⌛';
            document.getElementById('modalTitle').innerText    = "Time's Up!";
            document.getElementById('modalSubtitle').innerText = `The ${this.config.timerDuration}-second clock ran out! Great effort!`;
        } else {
            document.getElementById('modalIcon').innerText     = '🏁';
            document.getElementById('modalTitle').innerText    = 'Game Over';
            document.getElementById('modalSubtitle').innerText = 'Challenge finished!';
        }

        document.getElementById('modalScore').innerText = this.score;
        document.getElementById('modalBest').innerText  = isNewPersonalBest ? this.score : bestScore;
        document.getElementById('modalBestHolder').innerText = isNewPersonalBest ? '(New #1 Leader!)' : (bestName ? `by ${bestName}` : '');

        const nameInputContainer = document.getElementById('nameInputContainer');
        if (qualifiesForLeaderboard) {
            nameInputContainer.classList.add('active');
            nameInputContainer.querySelector('label').innerText = isNewPersonalBest
                ? '🎉 NEW #1 HIGH SCORE RECORD!'
                : '🏆 YOU ENTERED THE TOP 5 LEADERBOARD!';
            document.getElementById('playerNameInput').value = '';
            document.getElementById('playerNameInput').focus();
        } else {
            nameInputContainer.classList.remove('active');
        }

        document.getElementById('gameOverModal').classList.add('active');
    }

    // ── Save High Score (localStorage) ───────────────────────────────────────
    saveHighScoreName() {
        const raw        = document.getElementById('playerNameInput').value.trim();
        const playerName = raw || 'Champion';
        const today      = new Date();
        const dateStr    = `${String(today.getMonth()+1).padStart(2,'0')}/${String(today.getDate()).padStart(2,'0')}/${today.getFullYear()}`;

        // Load existing leaderboard, insert new entry, sort, keep top 5
        const lb = lsGet(LS_LB_KEY(this.currentGame), []);
        lb.push({ name: playerName, score: this.score, date: dateStr });
        lb.sort((a, b) => b.score - a.score || 0);
        const top5 = lb.slice(0, 5);
        lsSet(LS_LB_KEY(this.currentGame), top5);

        // Update modal
        document.getElementById('nameInputContainer').classList.remove('active');
        if (top5.length > 0) {
            document.getElementById('modalBest').innerText      = top5[0].score;
            document.getElementById('modalBestHolder').innerText = `by ${top5[0].name}`;
        }

        // Refresh high score badges
        this.highScores = this._computeHighScores();
        this.updateHighScoreBadges();
        audio.correct();
    }

    // ── Admin Panel ───────────────────────────────────────────────────────────
    openAdminModal() {
        audio.click();
        document.getElementById('adminLockState').style.display     = 'block';
        document.getElementById('adminDashboardState').style.display = 'none';
        document.getElementById('adminCard').classList.remove('wide');
        document.getElementById('adminPasscode').value = '';
        document.getElementById('adminModal').classList.add('active');
    }

    closeAdminModal() {
        audio.click();
        document.getElementById('adminModal').classList.remove('active');
        this.hideResetConfirm();
    }

    authenticateAdmin() {
        const code    = document.getElementById('adminPasscode').value;
        const correct = lsGet(LS_PASSCODE, '1234');
        // Allow actual code, master 'admin', or blank if default still active
        const ok = code === correct || code === 'admin' || (correct === '1234' && code === '');
        if (ok) {
            this.adminPasscode = code;
            audio.correct();
            document.getElementById('adminLockState').style.display     = 'none';
            document.getElementById('adminDashboardState').style.display = 'block';
            document.getElementById('adminCard').classList.add('wide');
            this.loadConfigIntoPanel();
        } else {
            audio.wrong();
            alert('Incorrect passcode!');
        }
    }

    loadConfigIntoPanel() {
        document.getElementById('cfgTimer').value        = this.config.timerDuration;
        document.getElementById('cfgTimerVal').innerText = this.config.timerDuration;
        document.getElementById('cfgAdaptive').checked  = this.config.adaptiveDifficulty;
        document.getElementById('cfgCheat').checked     = this.config.cheatMode;
        document.getElementById('cfgWordGrid').value        = this.config.wordFindSize;
        document.getElementById('cfgWordGridVal').innerText = this.config.wordFindSize;

        document.getElementById('cfgOpAdd').checked = this.config.mathOperators.includes('+');
        document.getElementById('cfgOpSub').checked = this.config.mathOperators.includes('-');
        document.getElementById('cfgOpMul').checked = this.config.mathOperators.includes('×');
        document.getElementById('cfgOpDiv').checked = this.config.mathOperators.includes('÷');

        document.getElementById('cfgNewPasscode').value = '';
    }

    saveAndCloseAdminPanel() {
        audio.click();
        this.config.timerDuration      = parseInt(document.getElementById('cfgTimer').value);
        this.config.adaptiveDifficulty = document.getElementById('cfgAdaptive').checked;
        this.config.cheatMode          = document.getElementById('cfgCheat').checked;
        this.config.wordFindSize       = parseInt(document.getElementById('cfgWordGrid').value);

        const ops = [];
        if (document.getElementById('cfgOpAdd').checked) ops.push('+');
        if (document.getElementById('cfgOpSub').checked) ops.push('-');
        if (document.getElementById('cfgOpMul').checked) ops.push('×');
        if (document.getElementById('cfgOpDiv').checked) ops.push('÷');
        this.config.mathOperators = ops.length > 0 ? ops : ['+'];

        this._saveConfig();
        this.closeAdminModal();
    }

    saveNewPasscode() {
        const pass = document.getElementById('cfgNewPasscode').value.trim();
        if (pass.length > 0) {
            lsSet(LS_PASSCODE, pass);
            audio.correct();
            alert('Admin passcode successfully updated!');
            document.getElementById('cfgNewPasscode').value = '';
        } else {
            audio.wrong();
            alert('Passcode cannot be empty!');
        }
    }

    switchAdminTab(tabName) {
        audio.click();
        document.querySelectorAll('#adminDashboardState .admin-tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('#adminDashboardState .admin-tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById(`tab-${tabName}`).classList.add('active');
        document.querySelectorAll('#adminDashboardState .admin-tab-btn').forEach(btn => {
            if (btn.innerText.toLowerCase().includes(tabName.slice(0, 3))) btn.classList.add('active');
        });
    }

    // ── Sandbox Controls ──────────────────────────────────────────────────────
    sandboxAddTime(seconds) {
        if (!this.currentGame || this.timeLeft <= 0) { alert('No active challenge running!'); return; }
        audio.click();
        this.timeLeft = Math.max(0, this.timeLeft + seconds);
        this.updateTimerUI();
    }

    sandboxAddScore(pts) {
        if (!this.currentGame) { alert('No active challenge running!'); return; }
        audio.click();
        this.addScore(pts);
    }

    // ── Reset Leaderboards ────────────────────────────────────────────────────
    showResetConfirm() {
        audio.click();
        document.getElementById('resetStep1').style.display = 'none';
        document.getElementById('resetStep2').style.display = 'block';
    }

    hideResetConfirm() {
        const s1 = document.getElementById('resetStep1');
        const s2 = document.getElementById('resetStep2');
        if (s1) s1.style.display = 'block';
        if (s2) s2.style.display = 'none';
    }

    executeReset() {
        audio.click();
        const code    = this.adminPasscode || document.getElementById('adminPasscode').value;
        const correct = lsGet(LS_PASSCODE, '1234');
        const ok = code === correct || code === 'admin' || (correct === '1234' && code === '');
        if (ok) {
            GAMES.forEach(g => localStorage.removeItem(LS_LB_KEY(g)));
            this.highScores = this._computeHighScores();
            this.updateHighScoreBadges();
            this.hideResetConfirm();
            audio.correct();
            this.closeAdminModal();
            alert('All leaderboard records have been wiped!');
        } else {
            audio.wrong();
            alert('Unauthorized. Check your admin passcode.');
            this.hideResetConfirm();
        }
    }

    // ── Leaderboard Modal ─────────────────────────────────────────────────────
    openLeaderboardsModal() {
        audio.click();
        document.getElementById('leaderboardsModal').classList.add('active');
        this.switchLeaderboardTab('color');
    }

    closeLeaderboardsModal() {
        audio.click();
        document.getElementById('leaderboardsModal').classList.remove('active');
    }

    switchLeaderboardTab(gameType) {
        audio.click();
        this.activeLeaderboardTab = gameType;

        document.querySelectorAll('#leaderboardTabs .admin-tab-btn').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('#leaderboardTabs .admin-tab-btn').forEach(btn => {
            if (btn.innerText.toLowerCase().includes(gameType)) btn.classList.add('active');
        });

        this.refreshLeaderboardTable(gameType);
    }

    refreshLeaderboardTable(gameType) {
        const container = document.getElementById('leaderboardTableContainer');
        const list      = lsGet(LS_LB_KEY(gameType), []);

        if (list.length === 0) {
            container.innerHTML = `<div class="no-records">No rankings recorded yet. Be the first!</div>`;
            return;
        }

        let html = `
            <table class="leaderboard-table">
                <thead><tr>
                    <th>Rank</th><th>Player</th>
                    <th style="text-align:right">Score</th>
                    <th style="text-align:right">Date</th>
                </tr></thead>
                <tbody>
        `;
        list.forEach((item, i) => {
            const cls = i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : i === 2 ? 'rank-3' : '';
            html += `<tr>
                <td class="leaderboard-rank ${cls}">#${i + 1}</td>
                <td class="leaderboard-name">${item.name}</td>
                <td class="leaderboard-score">${item.score}</td>
                <td class="leaderboard-date">${item.date || ''}</td>
            </tr>`;
        });
        html += `</tbody></table>`;
        container.innerHTML = html;
    }
}

// Initialize
const gameApp = new App();
