/* ==========================================
   MAIN APP ENGINE & STATE MANAGEMENT
   ========================================== */
class App {
    constructor() {
        this.currentGame = null;
        this.score = 0;
        this.timeLeft = 60;
        this.timerInterval = null;
        this.activeLeaderboardTab = 'color';

        // Bootstrap from data injected by Python at page-build time
        const initData = window.__STREAMLIT_INIT__ || {};
        this.config = initData.config || {
            timerDuration: 60,
            adaptiveDifficulty: true,
            wordFindSize: 8,
            mathOperators: ['+', '-', '\u00d7'],
            mathMaxRange: 50,
            cheatMode: false
        };
        this.highScores  = initData.highScores  || {};
        this.leaderboards = initData.leaderboards || {};
        this.adminPasscode = '';

        // Start initialization
        this.init();
    }

    async init() {
        // Apply initial high score badges right away (no round-trip needed)
        this.updateHighScoreBadges();
        this.updateAudioToggleUI();
    }

    sendRequest(action, payload = {}) {
        // Delegate to the bridge defined in the inline <script> block
        return window.sendToStreamlit(action, payload);
    }

    sendHeight() {
        const h = Math.max(document.documentElement.scrollHeight, 750);
        window.parent.postMessage({
            isStreamlitMessage: true,
            type: "streamlit:setFrameHeight",
            height: h
        }, "*");
    }

    async loadConfig() {
        // Automatically handled via Streamlit incoming messages
    }

    async loadScoresAndBadges() {
        // Automatically handled via Streamlit incoming messages
    }

    updateHighScoreBadges() {
        const games = ['color', 'word', 'memory', 'math', 'rule', 'chrono'];
        games.forEach(game => {
            const badgeEl = document.getElementById(`hs-${game}`);
            if (badgeEl && this.highScores[game]) {
                const score = this.highScores[game].score;
                const holder = this.highScores[game].holder;
                badgeEl.innerText = holder ? `${score} (${holder})` : score;
            }
        });
    }

    startPolling() {
        // Polling disabled in Streamlit since Python triggers updates Reactively
    }

    toggleAudio() {
        audio.enabled = !audio.enabled;
        this.updateAudioToggleUI();
    }

    updateAudioToggleUI() {
        const btn = document.getElementById('audioToggle');
        if (btn) {
            btn.innerText = audio.enabled ? '🔊' : '🔇';
        }
    }

    showHub() {
        this.stopTimer();
        document.getElementById('stageScreen').classList.remove('active');
        document.getElementById('hubScreen').classList.add('active');
        document.getElementById('gameOverModal').classList.remove('active');
        document.getElementById('adminModal').classList.remove('active');
        document.getElementById('leaderboardsModal').classList.remove('active');
        this.updateHighScoreBadges();
        this.currentGame = null;
        this.sendHeight();
    }

    startGame(gameType) {
        audio.click();
        this.currentGame = gameType;
        this.score = 0;
        this.timeLeft = this.config.timerDuration;
        document.getElementById('currentScore').innerText = '0';

        document.getElementById('hubScreen').classList.remove('active');
        document.getElementById('stageScreen').classList.add('active');
        document.getElementById('gameOverModal').classList.remove('active');
        document.getElementById('adminModal').classList.remove('active');
        document.getElementById('leaderboardsModal').classList.remove('active');

        const arena = document.getElementById('gameArena');
        arena.innerHTML = '';

        if (gameType === 'color') this.gameLogic = new ColorGame(arena, this);
        else if (gameType === 'word') this.gameLogic = new WordGame(arena, this);
        else if (gameType === 'memory') this.gameLogic = new MemoryGame(arena, this);
        else if (gameType === 'math') this.gameLogic = new MathGame(arena, this);
        else if (gameType === 'rule') this.gameLogic = new RuleGame(arena, this);
        else if (gameType === 'chrono') this.gameLogic = new ChronoGame(arena, this);

        this.gameLogic.init();
        this.startTimer();
        this.sendHeight();
    }

    restartCurrentGame() {
        if (this.currentGame) this.startGame(this.currentGame);
    }

    addScore(pts) {
        this.score += pts;
        document.getElementById('currentScore').innerText = this.score;
    }

    startTimer() {
        this.stopTimer();
        this.updateTimerUI();
        this.timerInterval = setInterval(() => {
            this.timeLeft--;
            this.updateTimerUI();
            if (this.timeLeft <= 5 && this.timeLeft > 0) audio.tick();
            if (this.timeLeft <= 0) {
                this.endGame(true);
            }
        }, 1000);
    }

    stopTimer() {
        if (this.timerInterval) clearInterval(this.timerInterval);
    }

    updateTimerUI() {
        document.getElementById('timerText').innerText = this.timeLeft;
        const circle = document.getElementById('timerProgress');
        const circumference = 2 * Math.PI * 35;
        const totalDuration = this.config.timerDuration;
        const offset = circumference - (this.timeLeft / totalDuration) * circumference;
        circle.style.strokeDashoffset = offset;
        if (this.timeLeft <= 10) {
            circle.style.stroke = '#ef4444';
        } else {
            circle.style.stroke = 'var(--accent-cyan)';
        }
    }

    async endGame(timeIsUp = true) {
        this.stopTimer();
        audio.levelUp();

        // Query local leaderboard rankings to determine placement qualification
        let qualifiesForLeaderboard = false;
        let isNewPersonalBest = false;
        let bestScore = 0;
        let bestName = '';

        const list = (this.leaderboards && this.leaderboards[this.currentGame]) || [];
        bestScore = list.length > 0 ? list[0].score : 0;
        bestName = list.length > 0 ? list[0].name : '';
        
        isNewPersonalBest = list.length === 0 || this.score > bestScore;
        qualifiesForLeaderboard = list.length < 5 || this.score > (list[list.length - 1]?.score || 0);

        if (timeIsUp) {
            document.getElementById('modalIcon').innerText = '⌛';
            document.getElementById('modalTitle').innerText = 'Time\'s Up!';
            document.getElementById('modalSubtitle').innerText = `The ${this.config.timerDuration}-second clock ran out! Great effort!`;
        } else {
            document.getElementById('modalIcon').innerText = '🏁';
            document.getElementById('modalTitle').innerText = 'Game Over';
            document.getElementById('modalSubtitle').innerText = 'Challenge finished!';
        }

        document.getElementById('modalScore').innerText = this.score;
        document.getElementById('modalBest').innerText = isNewPersonalBest ? this.score : bestScore;

        let leaderText = '';
        if (isNewPersonalBest) {
            leaderText = '(New #1 Leader!)';
        } else if (bestName) {
            leaderText = `by ${bestName}`;
        }
        document.getElementById('modalBestHolder').innerText = leaderText;

        const nameInputContainer = document.getElementById('nameInputContainer');
        if (qualifiesForLeaderboard) {
            nameInputContainer.classList.add('active');
            nameInputContainer.querySelector('label').innerText = isNewPersonalBest ? '🎉 NEW #1 HIGH SCORE RECORD!' : '🏆 YOU ENTERED THE TOP 5 LEADERBOARD!';
            document.getElementById('playerNameInput').value = '';
            document.getElementById('playerNameInput').focus();
        } else {
            nameInputContainer.classList.remove('active');
        }

        document.getElementById('gameOverModal').classList.add('active');
        this.sendHeight();
    }

    async saveHighScoreName() {
        const nameInput = document.getElementById('playerNameInput').value.trim();
        const playerName = nameInput || 'Champion';

        try {
            const data = await this.sendRequest('post_score', {
                game_type: this.currentGame,
                player_name: playerName,
                score: this.score
            });

            if (data && data.status === 'success') {
                document.getElementById('nameInputContainer').classList.remove('active');
                
                // Show updated leaderboard stats on modal best holder
                const list = data.leaderboard;
                if (list && list.length > 0) {
                    document.getElementById('modalBest').innerText = list[0].score;
                    document.getElementById('modalBestHolder').innerText = `by ${list[0].name}`;
                }
                
                audio.correct();
            }
        } catch (e) {
            console.error("Failed to save score:", e);
            alert("Connection error: Failed to save score.");
        }
    }

    /* ADMIN CONFIGURATOR & DEVELOPER PANEL */
    openAdminModal() {
        audio.click();
        document.getElementById('adminLockState').style.display = 'block';
        document.getElementById('adminDashboardState').style.display = 'none';
        document.getElementById('adminCard').classList.remove('wide');
        document.getElementById('adminPasscode').value = '';
        document.getElementById('adminModal').classList.add('active');
        this.sendHeight();
    }

    closeAdminModal() {
        audio.click();
        document.getElementById('adminModal').classList.remove('active');
        // Always reset the confirmation panel when closing
        this.hideResetConfirm();
        this.sendHeight();
    }

    async authenticateAdmin() {
        const code = document.getElementById('adminPasscode').value;
        try {
            const data = await this.sendRequest('authenticate_admin', { passcode: code });
            if (data && data.authenticated) {
                this.adminPasscode = code;
                audio.correct();
                document.getElementById('adminLockState').style.display = 'none';
                document.getElementById('adminDashboardState').style.display = 'block';
                document.getElementById('adminCard').classList.add('wide');
                
                this.loadConfigIntoPanel();
            } else {
                audio.wrong();
                alert('Incorrect passcode!');
            }
        } catch (e) {
            console.error("Authentication error:", e);
        }
    }

    loadConfigIntoPanel() {
        document.getElementById('cfgTimer').value = this.config.timerDuration;
        document.getElementById('cfgTimerVal').innerText = this.config.timerDuration;
        document.getElementById('cfgAdaptive').checked = this.config.adaptiveDifficulty;
        document.getElementById('cfgCheat').checked = this.config.cheatMode;
        document.getElementById('cfgWordGrid').value = this.config.wordFindSize;
        document.getElementById('cfgWordGridVal').innerText = this.config.wordFindSize;

        document.getElementById('cfgOpAdd').checked = this.config.mathOperators.includes('+');
        document.getElementById('cfgOpSub').checked = this.config.mathOperators.includes('-');
        document.getElementById('cfgOpMul').checked = this.config.mathOperators.includes('×');
        document.getElementById('cfgOpDiv').checked = this.config.mathOperators.includes('÷');
        
        document.getElementById('cfgNewPasscode').value = '';
        this.sendHeight();
    }

    async saveAndCloseAdminPanel() {
        audio.click();
        this.config.timerDuration = parseInt(document.getElementById('cfgTimer').value);
        this.config.adaptiveDifficulty = document.getElementById('cfgAdaptive').checked;
        this.config.cheatMode = document.getElementById('cfgCheat').checked;
        this.config.wordFindSize = parseInt(document.getElementById('cfgWordGrid').value);

        const ops = [];
        if (document.getElementById('cfgOpAdd').checked) ops.push('+');
        if (document.getElementById('cfgOpSub').checked) ops.push('-');
        if (document.getElementById('cfgOpMul').checked) ops.push('×');
        if (document.getElementById('cfgOpDiv').checked) ops.push('÷');
        this.config.mathOperators = ops.length > 0 ? ops : ['+'];

        try {
            const data = await this.sendRequest('update_config', { config: this.config });
            if (data && data.status === 'success') {
                this.closeAdminModal();
            }
        } catch (e) {
            console.error("Failed to save config:", e);
            alert("Failed to save configuration.");
        }
    }

    async saveNewPasscode() {
        const pass = document.getElementById('cfgNewPasscode').value.trim();
        if (pass.length > 0) {
            try {
                const data = await this.sendRequest('update_passcode', { passcode: pass });
                if (data && data.status === 'success') {
                    this.adminPasscode = pass;
                    document.getElementById('adminPasscode').value = pass;
                    audio.correct();
                    alert('Admin passcode successfully updated!');
                }
            } catch (e) {
                console.error(e);
            }
        } else {
            audio.wrong();
            alert('Passcode cannot be empty!');
        }
    }

    switchAdminTab(tabName) {
        audio.click();
        const contents = document.querySelectorAll('#adminDashboardState .admin-tab-content');
        const tabs = document.querySelectorAll('#adminDashboardState .admin-tab-btn');
        
        contents.forEach(el => el.classList.remove('active'));
        tabs.forEach(el => el.classList.remove('active'));

        document.getElementById(`tab-${tabName}`).classList.add('active');
        
        tabs.forEach(btn => {
            if (btn.innerText.toLowerCase().includes(tabName.slice(0,3))) {
                btn.classList.add('active');
            }
        });
        this.sendHeight();
    }

    /* DEV SANDBOX CONTROLS */
    sandboxAddTime(seconds) {
        if (!this.currentGame || this.timeLeft <= 0) {
            alert('No active challenge running!');
            return;
        }
        audio.click();
        this.timeLeft = Math.max(0, this.timeLeft + seconds);
        this.updateTimerUI();
    }

    sandboxAddScore(pts) {
        if (!this.currentGame) {
            alert('No active challenge running!');
            return;
        }
        audio.click();
        this.addScore(pts);
    }

    showResetConfirm() {
        audio.click();
        document.getElementById('resetStep1').style.display = 'none';
        document.getElementById('resetStep2').style.display = 'block';
        this.sendHeight();
    }

    hideResetConfirm() {
        const s1 = document.getElementById('resetStep1');
        const s2 = document.getElementById('resetStep2');
        if (s1) s1.style.display = 'block';
        if (s2) s2.style.display = 'none';
        this.sendHeight();
    }

    async executeReset() {
        audio.click();
        const adminCode = this.adminPasscode || document.getElementById('adminPasscode').value;
        try {
            const data = await this.sendRequest('reset_leaderboards', { passcode: adminCode });
            if (data && data.status === 'success') {
                this.hideResetConfirm();
                audio.correct();
                this.closeAdminModal();
                alert('✅ All leaderboard records have been wiped successfully!');
            } else {
                audio.wrong();
                alert('❌ Reset failed: ' + ((data && data.message) || 'Unauthorized. Check your admin passcode.'));
                this.hideResetConfirm();
            }
        } catch (e) {
            audio.wrong();
            console.error('Reset error:', e);
            alert('❌ Network error during reset. Is the server running?');
            this.hideResetConfirm();
        }
    }

    /* LEADERBOARD VIEW MODAL */
    async openLeaderboardsModal() {
        audio.click();
        document.getElementById('leaderboardsModal').classList.add('active');
        await this.switchLeaderboardTab('color');
        this.sendHeight();
    }

    closeLeaderboardsModal() {
        audio.click();
        document.getElementById('leaderboardsModal').classList.remove('active');
        this.sendHeight();
    }

    async switchLeaderboardTab(gameType) {
        audio.click();
        this.activeLeaderboardTab = gameType;
        
        const tabs = document.querySelectorAll('#leaderboardTabs .admin-tab-btn');
        tabs.forEach(el => el.classList.remove('active'));

        tabs.forEach(btn => {
            if (btn.innerText.toLowerCase().includes(gameType)) {
                btn.classList.add('active');
            }
        });

        await this.refreshLeaderboardTable(gameType);
    }

    async refreshLeaderboardTable(gameType) {
        const container = document.getElementById('leaderboardTableContainer');
        try {
            const list = (this.leaderboards && this.leaderboards[gameType]) || [];
            
            if (list.length === 0) {
                container.innerHTML = `<div class="no-records">No rankings recorded yet. Be the first!</div>`;
                return;
            }

            let html = `
                <table class="leaderboard-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Player</th>
                            <th style="text-align: right;">Score</th>
                            <th style="text-align: right;">Date</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            list.forEach((item, index) => {
                const rankClass = index === 0 ? 'rank-1' : index === 1 ? 'rank-2' : index === 2 ? 'rank-3' : '';
                html += `
                    <tr>
                        <td class="leaderboard-rank ${rankClass}">#${index + 1}</td>
                        <td class="leaderboard-name">${item.name}</td>
                        <td class="leaderboard-score">${item.score}</td>
                        <td class="leaderboard-date">${item.date}</td>
                    </tr>
                `;
            });

            html += `
                    </tbody>
                </table>
            `;
            container.innerHTML = html;
        } catch (e) {
            console.error("Failed to refresh leaderboard:", e);
        }
    }
}

// Initialize App Global
const gameApp = new App();
