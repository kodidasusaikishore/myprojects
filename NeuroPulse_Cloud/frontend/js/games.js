/* ==========================================
   GAME 1: COLOR MATCH (STROOP CHALLENGE)
   ========================================== */
class ColorGame {
    constructor(container, app) {
        this.container = container;
        this.app = app;
        this.colors = [
            { name: 'RED', hex: '#ef4444' },
            { name: 'BLUE', hex: '#3b82f6' },
            { name: 'GREEN', hex: '#10b981' },
            { name: 'YELLOW', hex: '#f59e0b' },
            { name: 'PURPLE', hex: '#a855f7' },
            { name: 'CYAN', hex: '#06b6d4' },
            { name: 'ORANGE', hex: '#f97316' },
            { name: 'PINK', hex: '#ec4899' },
            { name: 'WHITE', hex: '#f8fafc' },
            { name: 'LIME', hex: '#84cc16' }
        ];
    }

    init() {
        this.container.innerHTML = `
            <div class="color-container">
                <div class="color-instruction">Select the <span>INK COLOR</span> of the word below!</div>
                <div class="color-card" id="colorCard">RED</div>
                <div class="color-options" id="colorOptions"></div>
            </div>
        `;
        this.generateProblem();
    }

    generateProblem() {
        const textObj = this.colors[Math.floor(Math.random() * this.colors.length)];
        let inkObj = this.colors[Math.floor(Math.random() * this.colors.length)];

        if (Math.random() < 0.7 && inkObj.name === textObj.name) {
            const diffColors = this.colors.filter(c => c.name !== textObj.name);
            inkObj = diffColors[Math.floor(Math.random() * diffColors.length)];
        }

        this.correctInk = inkObj.name;
        const cardEl = document.getElementById('colorCard');
        cardEl.innerText = textObj.name;
        cardEl.style.color = inkObj.hex;

        // Adaptive option scaling based on progression score
        let optionsCount = 6;
        if (this.app.config.adaptiveDifficulty) {
            if (this.app.score < 50) optionsCount = 3;
            else if (this.app.score < 100) optionsCount = 4;
            else if (this.app.score < 180) optionsCount = 6;
            else optionsCount = 8;
        }

        let choicesSet = new Set([inkObj]);
        while (choicesSet.size < optionsCount) {
            let randomC = this.colors[Math.floor(Math.random() * this.colors.length)];
            choicesSet.add(randomC);
        }

        const choicesArr = Array.from(choicesSet).sort(() => 0.5 - Math.random());
        const optContainer = document.getElementById('colorOptions');
        optContainer.innerHTML = '';

        // Dynamically modify grid layout columns based on choice counts
        const cols = optionsCount <= 4 ? 2 : optionsCount <= 6 ? 3 : 4;
        optContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

        choicesArr.forEach(c => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'color-opt-btn';
            btn.innerHTML = `<span class="color-dot" style="background: ${c.hex};"></span> ${c.name}`;
            
            // Sandbox Cheat Mode support: highlight correct color option
            if (this.app.config.cheatMode && c.name === this.correctInk) {
                btn.style.outline = '2px solid var(--accent-cyan)';
            }

            btn.onclick = () => this.handleChoice(c.name);
            optContainer.appendChild(btn);
        });
    }

    handleChoice(selectedName) {
        if (selectedName === this.correctInk) {
            audio.correct();
            this.app.addScore(15);
        } else {
            audio.wrong();
        }
        this.generateProblem();
    }
}

/* ==========================================
   GAME 2: WORD FIND (8-WAY)
   ========================================== */
class WordGame {
    constructor(container, app) {
        this.container = container;
        this.app = app;
        this.level = 1;
        this.wordBank = {
            3: ['CAT', 'DOG', 'SUN', 'BOX', 'TOY', 'RED', 'MAP', 'PEN', 'FLY', 'KEY', 'JAM', 'CUB', 'BAT', 'ICE', 'SKY', 'RUN', 'WIN', 'FIT', 'FOX', 'GEM', 'HOT', 'PIN', 'NET', 'ZIP', 'RAY', 'OAK', 'OWL', 'MUD', 'NEW', 'BUG', 'WEB', 'FIG', 'NUT', 'POT', 'TAX', 'TEN', 'USE', 'VAN', 'WET', 'YAK'],
            4: ['GAME', 'PLAY', 'FIND', 'WORD', 'GRID', 'MIND', 'FAST', 'CALM', 'EASY', 'ZEAL', 'GLOW', 'ZONE', 'WAVE', 'RUSH', 'BOLT', 'WIND', 'STAR', 'BLUE', 'ROSE', 'GOLD', 'LION', 'HAWK', 'WOLF', 'BEAR', 'TREE', 'LEAF', 'ROCK', 'SAND', 'COIN', 'NOTE', 'CODE', 'LAMP', 'DOOR', 'PATH', 'GATE', 'BOAT', 'SHIP', 'NEST', 'CLAY', 'FIRE', 'WIND', 'TALK', 'WALK', 'JUMP', 'KITE'],
            5: ['BRAIN', 'SMART', 'FOCUS', 'LIGHT', 'FLASH', 'SPACE', 'DRAFT', 'PULSE', 'SHARK', 'PLANT', 'CLOCK', 'SOUND', 'COLOR', 'STAGE', 'THEME', 'LOGIC', 'SHAPE', 'STREAK', 'CLEAN', 'LEADER', 'SCORE', 'AUDIO', 'TRACK', 'HAPPY', 'CRAZY', 'SHINE', 'DREAM', 'FLUTE', 'MAGIC', 'WATER', 'CHIPS', 'BLOCK', 'BRICK', 'CLOUD', 'FLAME', 'GRASS', 'STONE', 'STORM', 'CROWN', 'GIANT'],
            6: ['ACTION', 'ACTIVE', 'ANCHOR', 'BRAINY', 'CLEVER', 'DETAIL', 'ENERGY', 'ENGINE', 'FUTURE', 'IMPACT', 'MASTER', 'METHOD', 'MOTION', 'OBJECT', 'OPTION', 'PATTER', 'POCKET', 'PORTAL', 'PROMPT', 'PUZZLE', 'RANDOM', 'REASON', 'RECALL', 'RECORD', 'RHYTHM', 'SCREEN', 'SEARCH', 'SENSOR', 'SHADOW', 'SIGNAL', 'SOURCE', 'SYSTEM', 'TARGET', 'THEORY', 'UNIQUE', 'VECTOR', 'VISION', 'VOLUME', 'VORTEX', 'WINNER', 'WISDOM', 'WONDER', 'ZIPPER', 'MATRIX', 'MEMORY', 'MENTAL', 'GENIUS', 'NUMBER', 'RATING', 'ANALYST']
        };
    }

    init() {
        this.generateLevel();
    }

    generateLevel() {
        this.foundWords = new Set();
        this.selectedSequence = [];
        this.targetCoords = [];

        let wordLen, maxWords, cellFontSize;
        if (this.level === 1) {
            this.size = 4;
            wordLen = 3;
            maxWords = 2;
            cellFontSize = '1.6rem';
        } else if (this.level === 2) {
            this.size = 6;
            wordLen = 4;
            maxWords = 3;
            cellFontSize = '1.3rem';
        } else {
            this.size = this.app.config.wordFindSize || 8;
            wordLen = Math.random() > 0.5 ? 5 : 6;
            maxWords = 4;
            cellFontSize = '1rem';
        }

        let selectedWordsPool = [];
        if (this.level >= 3) {
            selectedWordsPool = [...this.wordBank[5], ...this.wordBank[6]];
        } else {
            selectedWordsPool = [...this.wordBank[wordLen]];
        }

        const shuffled = selectedWordsPool.sort(() => 0.5 - Math.random());
        this.grid = Array(this.size).fill(null).map(() => Array(this.size).fill(''));

        const dirs = [
            { r: 0, c: 1 }, { r: 1, c: 0 }, { r: 1, c: 1 }, { r: 1, c: -1 },
            { r: 0, c: -1 }, { r: -1, c: 0 }, { r: -1, c: -1 }, { r: -1, c: 1 }
        ];

        this.targetWords = [];

        for (let word of shuffled) {
            if (this.targetWords.length >= maxWords) break;

            let placed = false;
            let attempts = 0;
            while (!placed && attempts < 100) {
                attempts++;
                const dir = dirs[Math.floor(Math.random() * dirs.length)];
                const row = Math.floor(Math.random() * this.size);
                const col = Math.floor(Math.random() * this.size);

                let fits = true;
                for (let i = 0; i < word.length; i++) {
                    const r = row + dir.r * i;
                    const c = col + dir.c * i;
                    if (r < 0 || r >= this.size || c < 0 || c >= this.size) {
                        fits = false;
                        break;
                    }
                    if (this.grid[r][c] !== '' && this.grid[r][c] !== word[i]) {
                        fits = false;
                        break;
                    }
                }

                if (fits) {
                    const coords = [];
                    for (let i = 0; i < word.length; i++) {
                        const r = row + dir.r * i;
                        const c = col + dir.c * i;
                        this.grid[r][c] = word[i];
                        coords.push({ r, c });
                    }
                    this.targetCoords.push(...coords);
                    this.targetWords.push(word);
                    placed = true;
                }
            }
        }

        this.container.innerHTML = `
            <div class="word-container">
                <div class="word-targets" id="wordTargets">
                    ${this.targetWords.map(w => `<div class="target-word" id="tw-${w}">${w}</div>`).join('')}
                </div>
                <div class="word-grid" id="wordGrid" style="grid-template-columns: repeat(${this.size}, 1fr);"></div>
            </div>
        `;

        const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
        for (let r = 0; r < this.size; r++) {
            for (let c = 0; c < this.size; c++) {
                if (this.grid[r][c] === '') {
                    this.grid[r][c] = letters[Math.floor(Math.random() * letters.length)];
                }
            }
        }

        const gridEl = document.getElementById('wordGrid');
        for (let r = 0; r < this.size; r++) {
            for (let c = 0; c < this.size; c++) {
                const cell = document.createElement('div');
                cell.className = 'word-cell';
                cell.style.fontSize = cellFontSize;
                cell.innerText = this.grid[r][c];
                cell.dataset.r = r;
                cell.dataset.c = c;
                
                if (this.app.config.cheatMode && this.targetCoords.some(coord => coord.r === r && coord.c === c)) {
                    cell.classList.add('cheat-highlight');
                }

                cell.onclick = () => this.handleCellClick(cell, r, c);
                gridEl.appendChild(cell);
            }
        }
    }

    handleCellClick(cell, r, c) {
        audio.click();

        const existingIdx = this.selectedSequence.findIndex(item => item.r === r && item.c === c);
        if (existingIdx !== -1) {
            this.selectedSequence.splice(existingIdx, 1);
            cell.classList.remove('selected');
        } else {
            cell.classList.add('selected');
            this.selectedSequence.push({ r, c, char: this.grid[r][c], el: cell });
        }

        const formed = this.selectedSequence.map(item => item.char).join('');
        const reversed = formed.split('').reverse().join('');

        this.targetWords.forEach(w => {
            if ((w === formed || w === reversed) && !this.foundWords.has(w)) {
                audio.correct();
                this.foundWords.add(w);
                document.getElementById(`tw-${w}`).classList.add('found');

                this.selectedSequence.forEach(item => {
                    item.el.classList.remove('selected');
                    item.el.classList.add('found-cell');
                    item.el.classList.remove('cheat-highlight');
                });
                this.selectedSequence = [];
                this.app.addScore(35);

                if (this.foundWords.size === this.targetWords.length) {
                    audio.levelUp();
                    this.app.addScore(60);
                    this.level++;
                    setTimeout(() => this.generateLevel(), 300);
                }
            }
        });
    }
}

/* ==========================================
   GAME 3: MEMORY FLASH (MATRIX RECALL)
   ========================================== */
class MemoryGame {
    constructor(container, app) {
        this.container = container;
        this.app = app;
        this.gridSize = 3;
        this.flashCount = 3;
    }

    init() {
        if (!this.app.config.adaptiveDifficulty) {
            this.gridSize = 3;
            this.flashCount = 4; // Static standard layout
        } else {
            this.gridSize = 3;
            this.flashCount = 3;
        }
        this.renderGrid();
        this.startRound();
    }

    renderGrid() {
        this.container.innerHTML = `
            <div class="memory-container">
                <div class="memory-status" id="memStatus">Memorize the tiles!</div>
                <div class="memory-grid" id="memGrid" style="grid-template-columns: repeat(${this.gridSize}, 1fr);"></div>
            </div>
        `;

        const gridEl = document.getElementById('memGrid');
        gridEl.innerHTML = '';
        this.tiles = [];
        const total = this.gridSize * this.gridSize;

        for (let i = 0; i < total; i++) {
            const tile = document.createElement('div');
            tile.className = 'memory-tile';
            tile.dataset.idx = i;
            tile.onclick = () => this.handleTileClick(i, tile);
            gridEl.appendChild(tile);
            this.tiles.push(tile);
        }
    }

    startRound() {
        this.userPattern = [];
        this.acceptInput = false;
        document.getElementById('memStatus').innerText = 'Memorize tiles...';

        const total = this.gridSize * this.gridSize;
        const indices = Array.from({ length: total }, (_, i) => i).sort(() => 0.5 - Math.random());
        this.targetPattern = indices.slice(0, this.flashCount);

        setTimeout(() => {
            this.targetPattern.forEach(idx => this.tiles[idx].classList.add('flash'));
            audio.tick();

            // Sandbox Cheat Mode outline support
            if (this.app.config.cheatMode) {
                this.targetPattern.forEach(idx => this.tiles[idx].classList.add('cheat-outline'));
            }

            setTimeout(() => {
                this.tiles.forEach(t => t.classList.remove('flash'));
                document.getElementById('memStatus').innerText = 'Recall pattern!';
                this.acceptInput = true;
            }, 1000);
        }, 400);
    }

    handleTileClick(idx, tile) {
        if (!this.acceptInput || this.userPattern.includes(idx)) return;

        if (this.targetPattern.includes(idx)) {
            audio.correct();
            tile.classList.add('user-correct');
            tile.classList.remove('cheat-outline'); // Clear outline once solved
            this.userPattern.push(idx);
            this.app.addScore(15);

            if (this.userPattern.length === this.targetPattern.length) {
                this.acceptInput = false;
                audio.levelUp();
                this.app.addScore(30);

                // Progression logic (only triggers if adaptive is enabled)
                if (this.app.config.adaptiveDifficulty) {
                    if (this.flashCount < 7) this.flashCount++;
                    else if (this.gridSize < 4) { this.gridSize = 4; this.flashCount = 4; }
                    else if (this.gridSize === 4 && this.flashCount < 10) this.flashCount++;
                }

                setTimeout(() => {
                    this.renderGrid();
                    this.startRound();
                }, 500);
            }
        } else {
            audio.wrong();
            tile.classList.add('user-wrong');
            this.acceptInput = false;
            document.getElementById('memStatus').innerText = 'Oops! Try again...';
            setTimeout(() => {
                this.tiles.forEach(t => t.className = 'memory-tile');
                this.startRound();
            }, 800);
        }
    }
}

/* ==========================================
   GAME 4: SPEED MATH
   ========================================== */
class MathGame {
    constructor(container, app) {
        this.container = container;
        this.app = app;
        this.streak = 0;
    }

    init() {
        this.container.innerHTML = `
            <div class="math-container">
                <div class="streak-badge">🔥 Streak: <span id="mathStreak">0</span>x</div>
                <div class="equation-card" id="mathEq">12 + 15 = ?</div>
                <div class="math-options" id="mathOptions"></div>
            </div>
        `;
        this.generateProblem();
    }

    generateProblem() {
        // Configurable list of arithmetic operators
        let allowedOps = this.app.config.mathOperators || ['+', '-', '×'];
        
        // Adaptive difficulty streak multipliers: dynamically enable × and ÷
        if (this.app.config.adaptiveDifficulty) {
            if (this.app.score > 80 && !allowedOps.includes('×')) {
                allowedOps = [...allowedOps, '×'];
            }
            if (this.streak >= 4 && !allowedOps.includes('÷')) {
                allowedOps = [...allowedOps, '÷'];
            }
        }
        
        const op = allowedOps[Math.floor(Math.random() * allowedOps.length)];
        let a, b, ans;

        // Scale numeric range dynamically with score
        let maxRange = this.app.config.mathMaxRange || 50;
        if (this.app.config.adaptiveDifficulty) {
            maxRange = Math.min(150, 40 + Math.floor(this.app.score / 5));
        }

        if (op === '+') {
            a = Math.floor(Math.random() * (maxRange - 5)) + 5;
            b = Math.floor(Math.random() * (maxRange - 5)) + 5;
            ans = a + b;
        } else if (op === '-') {
            a = Math.floor(Math.random() * maxRange) + 20;
            b = Math.floor(Math.random() * a);
            ans = a - b;
        } else if (op === '×') {
            const multiMax = this.app.config.adaptiveDifficulty ? Math.min(20, 10 + Math.floor(this.app.score / 60)) : 12;
            a = Math.floor(Math.random() * (multiMax - 2)) + 2;
            b = Math.floor(Math.random() * (multiMax - 2)) + 2;
            ans = a * b;
        } else { // Division '÷'
            const divMax = this.app.config.adaptiveDifficulty ? Math.min(15, 10 + Math.floor(this.app.score / 80)) : 10;
            b = Math.floor(Math.random() * (divMax - 2)) + 2;
            ans = Math.floor(Math.random() * (divMax - 2)) + 2;
            a = ans * b; // guarantee integers
        }

        document.getElementById('mathEq').innerText = `${a} ${op} ${b} = ?`;

        let options = new Set([ans]);
        while (options.size < 4) {
            let offset = (Math.floor(Math.random() * 5) + 1) * (Math.random() > 0.5 ? 1 : -1);
            let wrong = ans + offset;
            if (wrong >= 0) options.add(wrong);
        }

        const optArr = Array.from(options).sort(() => 0.5 - Math.random());
        const optContainer = document.getElementById('mathOptions');
        optContainer.innerHTML = '';

        optArr.forEach(opt => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'math-opt-btn';
            btn.innerText = opt;
            
            // Sandbox Cheat Mode helper: highlight correct arithmetic answer
            if (this.app.config.cheatMode && opt === ans) {
                btn.style.outline = '2px solid var(--accent-cyan)';
            }

            btn.onclick = () => this.handleAnswer(opt, ans);
            optContainer.appendChild(btn);
        });
    }

    handleAnswer(selected, correct) {
        if (selected === correct) {
            audio.correct();
            this.streak++;
            const pts = 10 + (this.streak * 2);
            this.app.addScore(pts);
        } else {
            audio.wrong();
            this.streak = 0;
        }
        document.getElementById('mathStreak').innerText = this.streak;
        this.generateProblem();
    }
}

/* ==========================================
   GAME 5: RULE SWITCH (DUAL TASK EXECUTIVES)
   ========================================== */
class RuleGame {
    constructor(container, app) {
        this.container = container;
        this.app = app;
    }

    init() {
        this.container.innerHTML = `
            <div class="rule-container">
                <div class="rule-badge" id="ruleBadge">MATCH SHAPE</div>
                <div class="rule-target-card" id="ruleTarget">🔵</div>
                <div class="rule-options" id="ruleOptions"></div>
            </div>
        `;
        this.generateProblem();
    }

    generateProblem() {
        // Dynamic Rule mode
        this.mode = Math.random() > 0.5 ? 'SHAPE' : 'COLOR';
        const badgeEl = document.getElementById('ruleBadge');
        badgeEl.innerText = this.mode === 'SHAPE' ? '🎯 MATCH SHAPE!' : '🎨 MATCH COLOR!';
        badgeEl.className = `rule-badge ${this.mode === 'SHAPE' ? 'shape-mode' : 'color-mode'}`;

        // Clean pool of distinct shape & color objects (expanded shapes & colors for higher difficulty)
        const pool = [
            // Circles
            { icon: '🔵', shape: 'circle', color: 'blue' },
            { icon: '🔴', shape: 'circle', color: 'red' },
            { icon: '🟢', shape: 'circle', color: 'green' },
            { icon: '🟡', shape: 'circle', color: 'yellow' },
            { icon: '🟣', shape: 'circle', color: 'purple' },
            { icon: '🟠', shape: 'circle', color: 'orange' },
            // Squares
            { icon: '🟦', shape: 'square', color: 'blue' },
            { icon: '🟥', shape: 'square', color: 'red' },
            { icon: '🟩', shape: 'square', color: 'green' },
            { icon: '🟨', shape: 'square', color: 'yellow' },
            { icon: '🟪', shape: 'square', color: 'purple' },
            { icon: '🟧', shape: 'square', color: 'orange' },
            // Hearts
            { icon: '💙', shape: 'heart', color: 'blue' },
            { icon: '❤️', shape: 'heart', color: 'red' },
            { icon: '💚', shape: 'heart', color: 'green' },
            { icon: '💛', shape: 'heart', color: 'yellow' },
            { icon: '💜', shape: 'heart', color: 'purple' },
            { icon: '🧡', shape: 'heart', color: 'orange' },
            // Diamonds
            { icon: '🔷', shape: 'diamond', color: 'blue' },
            { icon: '🔶', shape: 'diamond', color: 'orange' }
        ];

        const target = pool[Math.floor(Math.random() * pool.length)];
        document.getElementById('ruleTarget').innerText = target.icon;

        // Candidates matching active rule (different icon)
        const validMatches = pool.filter(item => item.icon !== target.icon && (this.mode === 'SHAPE' ? item.shape === target.shape : item.color === target.color));
        const correctOpt = validMatches[Math.floor(Math.random() * validMatches.length)];

        // Candidates violating active rule (wrong options)
        const wrongCandidates = pool.filter(item => item.icon !== target.icon && (this.mode === 'SHAPE' ? item.shape !== target.shape : item.color !== target.color));
        wrongCandidates.sort(() => 0.5 - Math.random());
        const selectedWrong = wrongCandidates.slice(0, 5);

        // Always assemble exactly 6 distinct options
        const choices = [correctOpt, ...selectedWrong].sort(() => 0.5 - Math.random());
        const optContainer = document.getElementById('ruleOptions');
        optContainer.innerHTML = '';

        choices.forEach(item => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'rule-opt-btn';
            btn.innerText = item.icon;
            
            // Sandbox Cheat Mode support: highlight correct rule match
            if (this.app.config.cheatMode && item.icon === correctOpt.icon) {
                btn.style.outline = '2px solid var(--accent-cyan)';
            }

            btn.onclick = () => this.handleChoice(item, target);
            optContainer.appendChild(btn);
        });
    }

    handleChoice(selected, target) {
        let isCorrect = false;
        if (this.mode === 'SHAPE' && selected.shape === target.shape) isCorrect = true;
        if (this.mode === 'COLOR' && selected.color === target.color) isCorrect = true;

        if (isCorrect) {
            audio.correct();
            this.app.addScore(20);
        } else {
            audio.wrong();
        }
        this.generateProblem();
    }
}

/* ==========================================
   GAME 6: CHRONO RUSH (ASCENDING NUMBERS)
   ========================================== */
class ChronoGame {
    constructor(container, app) {
        this.container = container;
        this.app = app;
        this.currentIndex = 0;
    }

    init() {
        this.container.innerHTML = `
            <div class="chrono-container">
                <div class="chrono-instruction">Tap numbers from <span>LOWEST to HIGHEST</span>!</div>
                <div class="chrono-grid" id="chronoGrid"></div>
            </div>
        `;
        this.generateLevel();
    }

    generateLevel() {
        let numSet = new Set();
        while (numSet.size < 9) {
            numSet.add(Math.floor(Math.random() * 89) + 10);
        }
        const numbers = Array.from(numSet);
        this.sortedNumbers = [...numbers].sort((a, b) => a - b);
        this.currentIndex = 0;

        const gridEl = document.getElementById('chronoGrid');
        gridEl.innerHTML = '';

        numbers.forEach(num => {
            const tile = document.createElement('div');
            tile.className = 'chrono-tile';
            tile.innerText = num;
            
            // Sandbox Cheat Mode support: highlight next target number
            if (this.app.config.cheatMode && num === this.sortedNumbers[0]) {
                tile.style.outline = '2px solid var(--accent-cyan)';
            }

            tile.onclick = () => this.handleTileClick(tile, num);
            gridEl.appendChild(tile);
        });
    }

    handleTileClick(tile, num) {
        if (tile.classList.contains('tapped')) return;

        const targetNum = this.sortedNumbers[this.currentIndex];
        if (num === targetNum) {
            audio.correct();
            tile.classList.add('tapped');
            tile.style.outline = 'none'; // Clear cheat outline
            this.app.addScore(10);
            this.currentIndex++;

            if (this.currentIndex >= 9) {
                audio.levelUp();
                this.app.addScore(30);
                setTimeout(() => this.generateLevel(), 250);
            } else {
                // Cheat Mode support: highlight the next number
                if (this.app.config.cheatMode) {
                    const nextNum = this.sortedNumbers[this.currentIndex];
                    const tiles = document.querySelectorAll('.chrono-tile');
                    tiles.forEach(t => {
                        if (parseInt(t.innerText) === nextNum) {
                            t.style.outline = '2px solid var(--accent-cyan)';
                        } else {
                            t.style.outline = 'none';
                        }
                    });
                }
            }
        } else {
            audio.wrong();
        }
    }
}
