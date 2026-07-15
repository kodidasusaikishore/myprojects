/* ==========================================
   AUDIO ENGINE (Web Audio API Synthesizer)
   ========================================== */
class AudioEngine {
    constructor() {
        this.ctx = null;
        this.enabled = true;
    }

    init() {
        if (!this.ctx) {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        }
    }

    playTone(freq, type = 'sine', duration = 0.1, gainVal = 0.1) {
        if (!this.enabled) return;
        this.init();
        try {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
            gain.gain.setValueAtTime(gainVal, this.ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + duration);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start();
            osc.stop(this.ctx.currentTime + duration);
        } catch (e) { }
    }

    click() { this.playTone(600, 'triangle', 0.05, 0.05); }
    correct() {
        this.playTone(523.25, 'sine', 0.08, 0.1);
        setTimeout(() => this.playTone(659.25, 'sine', 0.12, 0.1), 80);
    }
    wrong() {
        this.playTone(220, 'sawtooth', 0.15, 0.15);
    }
    levelUp() {
        const notes = [440, 554.37, 659.25, 880];
        notes.forEach((n, i) => setTimeout(() => this.playTone(n, 'sine', 0.1, 0.1), i * 60));
    }
    tick() { this.playTone(800, 'sine', 0.03, 0.02); }
}

const audio = new AudioEngine();
