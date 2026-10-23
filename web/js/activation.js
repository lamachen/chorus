/**
 * Proaktiver Aktivierungs-Scorer (StreamBridge-Konzept, arXiv:2505.05467).
 *
 * Rein numerisch und rein client-seitig: betrachtet die echten Frames auf dem
 * Grab-Canvas (Neuheit ggü. Ankerframe, Bewegungsenergie) plus Gesprächspause
 * und Sprechzustände. Er entscheidet NUR, *wann* die Figur von selbst das
 * Wort ergreift - WAS sie sagt, generiert ausschließlich das On-Device-VLM.
 * Keine Texte, keine Antworten, keine Skripte in dieser Datei.
 */

const THUMB = 48;

export class Activation {
  constructor(opts = {}) {
    this.threshold = opts.threshold ?? 0.62;
    this.cooldown = opts.cooldown ?? 22;
    this.minLull = opts.minLull ?? 4;
    this.noveltyGain = opts.noveltyGain ?? 2.6;
    this.lullGain = opts.lullGain ?? 0.9;
    this.enabled = opts.enabled !== false;
    this._c = null;
    this.prev = null;
    this.anchor = null;
    this.motionEma = 0;
    this.lastFire = 0;
  }

  tune(o = {}) {
    if (o.threshold != null) this.threshold = +o.threshold;
    if (o.cooldown != null) this.cooldown = +o.cooldown;
    if (o.enabled != null) this.enabled = !!o.enabled;
  }

  reset() { this.prev = this.anchor = null; this.motionEma = 0; this.lastFire = 0; }

  _thumb(canvas) {
    if (!canvas || !canvas.width) return null;
    const c = this._c || (this._c = document.createElement('canvas'));
    c.width = THUMB; c.height = THUMB;
    const x = c.getContext('2d', { willReadFrequently: true });
    x.drawImage(canvas, 0, 0, THUMB, THUMB);
    return x.getImageData(0, 0, THUMB, THUMB).data;
  }

  static _diff(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i += 4) s += Math.abs(a[i] - b[i]);
    return Math.min(1, s / (a.length / 4) / 60);
  }

  /**
   * @param {HTMLCanvasElement} canvas  der exakte Modell-Frame
   * @param {object} s { now, userSpeaking, sheSpeaking, lastExchange }
   * @returns {{score:number, speak:boolean, parts:object, reason:string}}
   */
  observe(canvas, s) {
    const now = s.now;
    const th = this._thumb(canvas);
    if (!th) return { score: 0, speak: false, parts: {}, reason: 'kein Bild' };

    const motion = this.prev ? Activation._diff(th, this.prev) : 0;
    this.prev = th;
    this.motionEma = 0.85 * this.motionEma + 0.15 * motion;
    if (!this.anchor) this.anchor = th;
    const novelty = Activation._diff(th, this.anchor);
    const lull = s.lastExchange ? Math.max(0, (now - s.lastExchange) / 1000) : 0;

    if (!this.enabled || s.userSpeaking || s.sheSpeaking) {
      return { score: 0, speak: false, parts: { motion, novelty, lull }, reason: 'beschäftigt' };
    }

    const score = 1 / (1 + Math.exp(-(
      1.6 * Math.min(1, novelty * this.noveltyGain) +
      this.lullGain * Math.min(1, lull / 12) -
      0.35 * Math.min(1, this.motionEma * 6) - 0.9)));

    const cooled = (now - this.lastFire) / 1000 >= this.cooldown;
    const speak = score >= this.threshold && cooled && lull >= this.minLull;
    if (speak) { this.lastFire = now; this.anchor = th; }
    return {
      score, speak,
      parts: { motion: +motion.toFixed(3), novelty: +novelty.toFixed(3), lull: +lull.toFixed(1) },
      reason: speak ? 'spricht von selbst' : (cooled ? (lull >= this.minLull ? 'nichts Auffälliges' : 'zu wenig Pause') : 'Cooldown'),
    };
  }

  fire() { this.lastFire = performance.now(); this.anchor = this.prev; }
}
