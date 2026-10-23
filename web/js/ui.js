/* DOM helpers, phone-app chrome (tabs, keypad, contacts), settings sheet, toasts */

export const $ = (id) => document.getElementById(id);
export const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};

/* ---------------- status bar clock ---------------- */
export function startClock(node) {
  const tick = () => {
    const d = new Date();
    node.textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  tick();
  setInterval(tick, 15000);
}

/* ---------------- avatars ---------------- */
export function setHue(root, hue) { root.style.setProperty('--hue', hue); }

export function avatarNode(name, hue, big = false) {
  const a = el('div', 'avatar' + (big ? ' big' : ''));
  a.style.setProperty('--hue', hue);
  a.innerHTML = `<span>${(name || '?').slice(0, 1)}</span>`;
  return a;
}

/* ---------------- contacts / recents ---------------- */
export function renderContacts(list, personas, onCall) {
  list.innerHTML = '';
  personas.forEach((p) => {
    const li = el('li');
    li.appendChild(avatarNode(p.name, p.hue));
    const main = el('button', 'c-main');
    main.innerHTML = `<div class="c-name">${p.name}, ${p.age}</div>
      <div class="c-sub"><span class="on">●</span> ${p.tagline || 'Video-Chat'}</div>`;
    main.onclick = () => onCall(p);
    const call = el('button', 'c-call',
      `<svg viewBox="0 0 24 24" class="ic"><path d="M15 8.5l4-3.5v14l-4-3.5zM3 6h11v12H3z"/></svg>`);
    call.onclick = (e) => { e.stopPropagation(); onCall(p); };
    li.append(main, call);
    list.appendChild(li);
  });
}

export function renderRecents(list, empty, calls, onCall) {
  list.innerHTML = '';
  empty.style.display = calls.length ? 'none' : 'block';
  calls.slice(0, 25).forEach((c) => {
    const p = c.persona;
    const li = el('li');
    li.appendChild(avatarNode(p.name, p.hue));
    const main = el('div', 'c-main');
    main.innerHTML = `<div class="c-name">${p.name}</div>
      <div class="c-sub">${c.missed ? '<span style="color:#ff6b61">entgangen</span>' : 'Video-Chat · ausgehend'}</div>`;
    const meta = el('div', 'recent-meta', `${fmtDur(c.seconds)}<br>${fmtWhen(c.at)}`);
    const btn = el('button', 'c-call',
      `<svg viewBox="0 0 24 24" class="ic"><path d="M6.6 10.8c1.4 2.8 3.8 5.1 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.1.4 2.3.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1C10.6 21 3 13.4 3 4c0-.6.4-1 1-1h3.5c.6 0 1 .4 1 1 0 1.2.2 2.4.6 3.6.1.4 0 .8-.3 1z"/></svg>`);
    btn.onclick = () => onCall(p);
    li.append(main, meta, btn);
    list.appendChild(li);
  });
}

export function fmtDur(s) {
  s = Math.max(0, Math.round(s || 0));
  const m = Math.floor(s / 60), r = s % 60;
  return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`;
}
export function fmtWhen(ts) {
  const d = new Date(ts), now = new Date();
  if (d.toDateString() === now.toDateString()) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return d.toLocaleDateString([], { day: '2-digit', month: '2-digit' });
}

/* ---------------- keypad ---------------- */
const KEYS = [
  ['1', ''], ['2', 'ABC'], ['3', 'DEF'], ['4', 'GHI'], ['5', 'JKL'], ['6', 'MNO'],
  ['7', 'PQRS'], ['8', 'TUV'], ['9', 'WXYZ'], ['*', ''], ['0', '+'], ['#', ''],
];
export function renderKeypad(box, onKey) {
  box.innerHTML = '';
  KEYS.forEach(([d, l]) => {
    const b = el('button', 'key', `<b>${d}</b><small>${l}</small>`);
    b.onclick = () => onKey(d);
    box.appendChild(b);
  });
}

/* ---------------- tabs ---------------- */
export function wireTabs(tabs, title, titles) {
  tabs.forEach((t) => {
    t.onclick = () => {
      tabs.forEach((x) => x.classList.remove('active'));
      t.classList.add('active');
      const key = t.dataset.tab;
      document.querySelectorAll('.tabpane').forEach((p) => p.classList.remove('active'));
      const pane = $('pane-' + (key === 'recents' ? 'recents' : key));
      pane && pane.classList.add('active');
      title.textContent = titles[key] || 'Telefon';
    };
  });
}

/* ---------------- toast ---------------- */
let toastTimer = null;
export function toast(msg, ms = 3200) {
  const t = $('toast');
  t.innerHTML = msg;
  t.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('on'), ms);
}

/* ---------------- settings sheet ---------------- */
/**
 * schema: [ {group:'Titel'}, {type:'switch|range|select|text|info|button', key, label, desc, ...} ]
 */
export function buildSettings(container, schema, values, onChange) {
  container.innerHTML = '';
  schema.forEach((item) => {
    if (item.group) { container.appendChild(el('div', 'grp', item.group)); return; }
    if (item.type === 'warn') { container.appendChild(el('div', 'warnbox', item.html)); return; }
    const row = el('div', 'row');
    const lab = el('label', null, item.label || '');
    if (item.key) lab.setAttribute('for', 'set-' + item.key);
    row.appendChild(lab);

    let input;
    switch (item.type) {
      case 'switch': {
        const wrap = el('div', 'switch');
        input = el('input'); input.type = 'checkbox'; input.id = 'set-' + item.key;
        input.checked = !!values[item.key];
        const i = el('i');
        wrap.append(input, i);
        input.onchange = () => onChange(item.key, input.checked, item);
        row.appendChild(wrap);
        break;
      }
      case 'range': {
        input = el('input'); input.type = 'range'; input.id = 'set-' + item.key;
        Object.entries(item.attrs || {}).forEach(([k, v]) => input.setAttribute(k, v));
        input.value = values[item.key];
        const out = el('output', null, (item.fmt ? item.fmt(values[item.key]) : values[item.key]));
        input.oninput = () => {
          const v = item.step && String(item.attrs?.step).includes('.') ? parseFloat(input.value) : parseInt(input.value, 10);
          out.textContent = item.fmt ? item.fmt(v) : v;
          onChange(item.key, v, item);
        };
        row.append(input, out);
        break;
      }
      case 'select': {
        input = el('select'); input.id = 'set-' + item.key;
        (item.options || []).forEach((o) => {
          const op = el('option', null, o.label);
          op.value = o.value;
          if (String(o.value) === String(values[item.key])) op.selected = true;
          input.appendChild(op);
        });
        input.onchange = () => onChange(item.key, input.value, item);
        row.appendChild(input);
        break;
      }
      case 'text': {
        input = el('input'); input.type = item.password ? 'password' : 'text';
        input.id = 'set-' + item.key; input.value = values[item.key] ?? '';
        input.placeholder = item.placeholder || '';
        input.onchange = () => onChange(item.key, input.value, item);
        row.appendChild(input);
        break;
      }
      case 'button': {
        input = el('button', 'nav-btn', item.buttonLabel || 'Los');
        input.style.cssText += 'background:var(--blue);color:#fff;border-color:transparent;font-weight:600';
        input.onclick = () => onChange(item.key, true, item);
        row.appendChild(input);
        break;
      }
      case 'info': {
        const box = el('div'); box.style.flex = '1';
        box.innerHTML = (item.rows || []).map(([k, v]) =>
          `<div class="kv"><span>${k}</span><b>${v}</b></div>`).join('');
        box.dataset.infoKey = item.key || '';
        row.replaceChildren(box);
        container.appendChild(row);
        return;
      }
      default: return;
    }
    container.appendChild(row);
    if (item.desc) {
      const d = el('div', 'row');
      d.innerHTML = `<div class="desc">${item.desc}</div>`;
      d.style.padding = '0 0 8px';
      container.appendChild(d);
    }
  });
}

export function updateInfoBlock(container, key, rows) {
  const box = container.querySelector(`[data-info-key="${key}"]`);
  if (!box) return;
  box.innerHTML = rows.map(([k, v]) => `<div class="kv"><span>${k}</span><b>${v}</b></div>`).join('');
}

/* ---------------- DTMF (echte Zweifrequenz-Töne) ---------------- */
let _dtmfCtx = null;
const FREQS = { '1':[697,1209],'2':[697,1336],'3':[697,1477],'4':[770,1209],'5':[770,1336],'6':[770,1477],'7':[852,1209],'8':[852,1336],'9':[852,1477],'*':[941,1209],'0':[941,1336],'#':[941,1477] };
export function dtmf(d) {
  try {
    _dtmfCtx = _dtmfCtx || new (window.AudioContext || window.webkitAudioContext)();
    const f = FREQS[d]; if (!f) return;
    const t = _dtmfCtx.currentTime, g = _dtmfCtx.createGain();
    g.gain.setValueAtTime(0.09, t); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
    g.connect(_dtmfCtx.destination);
    f.forEach((hz) => { const o = _dtmfCtx.createOscillator(); o.frequency.value = hz; o.connect(g); o.start(t); o.stop(t + 0.17); });
  } catch {}
}

/* ---------------- ring tone (WebAudio, no assets needed) ---------------- */
export class RingTone {
  constructor() { this.ctx = null; this.timer = null; }
  start() {
    try {
      this.ctx = this.ctx || new (window.AudioContext || window.webkitAudioContext)();
      if (this.ctx.state === 'suspended') this.ctx.resume();
      const beep = () => {
        const t = this.ctx.currentTime;
        const o = this.ctx.createOscillator(), g = this.ctx.createGain();
        o.type = 'sine'; o.frequency.value = 425;             // German ring-back
        const o2 = this.ctx.createOscillator(); o2.type = 'sine'; o2.frequency.value = 425;
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(0.13, t + 0.04);
        g.gain.setValueAtTime(0.13, t + 0.85);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 1.0);
        o.connect(g); o2.connect(g); g.connect(this.ctx.destination);
        o.start(t); o2.start(t); o.stop(t + 1.05); o2.stop(t + 1.05);
        try { navigator.vibrate && navigator.vibrate([160, 90, 160]); } catch {}
      };
      beep();
      this.timer = setInterval(beep, 4000);
    } catch (e) { /* audio blocked - visual ringing still works */ }
  }
  stop() {
    clearInterval(this.timer); this.timer = null;
    try { this.ctx && this.ctx.close(); } catch {}
    this.ctx = null;
  }
}

/* ---------------- wake lock ---------------- */
export async function keepAwake(on) {
  try {
    if (on && 'wakeLock' in navigator) { window.__wake = await navigator.wakeLock.request('screen'); return true; }
    if (!on && window.__wake) { await window.__wake.release(); window.__wake = null; return true; }
  } catch (e) { /* not available */ }
  return false;
}
