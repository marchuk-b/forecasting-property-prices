/* Tiny dependency-free SVG charts: scatter, hbar, histogram, rangeBar. */
const Charts = (() => {
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  const full = (n) => Math.round(n).toLocaleString("en-US");

  function compact(n) {
    const a = Math.abs(n);
    const trim = (x) => x.toFixed(1).replace(/\.0$/, "");
    if (a >= 1e9) return trim(n / 1e9) + "B";
    if (a >= 1e6) return trim(n / 1e6) + "M";
    if (a >= 1e3) return trim(n / 1e3) + "k";
    return String(Math.round(n * 100) / 100);
  }

  function niceTicks(min, max, count = 5) {
    const span = max - min || 1;
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function put(el, svg) { el.classList.add("chart"); el.innerHTML = svg; }

  /* points: [[x, y], ...]; draws the y = x line (perfect prediction) */
  function scatter(el, pts, o = {}) {
    const W = 560, H = 380, m = { l: 62, r: 14, t: 10, b: 46 };
    const all = pts.flat();
    const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.04;
    const min = lo - pad, max = hi + pad;
    const sx = (v) => m.l + (v - min) / (max - min) * (W - m.l - m.r);
    const sy = (v) => H - m.b - (v - min) / (max - min) * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || "Scatter plot")}">`;
    niceTicks(min, max, 5).forEach((t) => {
      s += `<line class="c-grid" x1="${sx(t)}" x2="${sx(t)}" y1="${m.t}" y2="${H - m.b}"/>`;
      s += `<line class="c-grid" x1="${m.l}" x2="${W - m.r}" y1="${sy(t)}" y2="${sy(t)}"/>`;
      s += `<text class="c-text" x="${sx(t)}" y="${H - m.b + 16}" text-anchor="middle">${compact(t)}</text>`;
      s += `<text class="c-text" x="${m.l - 6}" y="${sy(t) + 4}" text-anchor="end">${compact(t)}</text>`;
    });
    s += `<line class="c-line" x1="${sx(min)}" y1="${sy(min)}" x2="${sx(max)}" y2="${sy(max)}"/>`;
    pts.forEach((p) => {
      s += `<circle class="c-dot" cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="3.5">` +
           `<title>actual ${full(p[0])}, predicted ${full(p[1])}</title></circle>`;
    });
    s += `<text class="c-text" x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xLabel || "")}</text>`;
    s += `<text class="c-text" transform="translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(o.yLabel || "")}</text>`;
    put(el, s + "</svg>");
  }

  /* items: [{label, value, err?}], bars grow from zero (negative values go left, red) */
  function hbar(el, items, o = {}) {
    const rowH = 26, W = 560, m = { l: o.labelW || 130, r: 64, t: 6, b: 6 };
    const H = m.t + m.b + items.length * rowH;
    const fmt = o.fmt || ((v) => v.toFixed(3));
    let min = Math.min(0, ...items.map((i) => i.value - (i.err || 0)));
    let max = Math.max(0, ...items.map((i) => i.value + (i.err || 0)));
    if (max === min) max = min + 1;
    const sx = (v) => m.l + (v - min) / (max - min) * (W - m.l - m.r);
    const x0 = sx(0);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || "Bar chart")}">`;
    items.forEach((it, i) => {
      const y = m.t + i * rowH, x = sx(it.value);
      s += `<text class="c-label" x="${m.l - 8}" y="${y + 17}" text-anchor="end">${esc(it.label)}</text>`;
      s += `<rect class="${it.value < 0 ? "c-neg" : "c-bar"}" x="${Math.min(x0, x).toFixed(1)}" y="${y + 4}" ` +
           `width="${Math.max(Math.abs(x - x0), 1).toFixed(1)}" height="${rowH - 9}" rx="2"/>`;
      let right = Math.max(x0, x);
      if (it.err) {
        const a = sx(it.value - it.err), b = sx(it.value + it.err), cy = y + rowH / 2;
        s += `<line class="c-err" x1="${a}" x2="${b}" y1="${cy}" y2="${cy}"/>`;
        right = Math.max(right, b);
      }
      s += `<text class="c-text" x="${right + 6}" y="${y + 17}">${esc(fmt(it.value))}</text>`;
    });
    s += `<line class="c-axis" x1="${x0}" x2="${x0}" y1="${m.t}" y2="${H - m.b}"/>`;
    put(el, s + "</svg>");
  }

  function histogram(el, edges, counts, o = {}) {
    const W = 560, H = 300, m = { l: 44, r: 14, t: 10, b: 44 };
    const min = edges[0], max = edges[edges.length - 1];
    const ytop = Math.max(...counts) * 1.08;      // a little headroom above the tallest bar
    const yt = niceTicks(0, ytop, 4);
    const sx = (v) => m.l + (v - min) / (max - min) * (W - m.l - m.r);
    const sy = (v) => H - m.b - v / ytop * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || "Histogram")}">`;
    yt.forEach((t) => {
      s += `<line class="c-grid" x1="${m.l}" x2="${W - m.r}" y1="${sy(t)}" y2="${sy(t)}"/>`;
      s += `<text class="c-text" x="${m.l - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
    });
    counts.forEach((c, i) => {
      const x = sx(edges[i]) + 1, w = Math.max(sx(edges[i + 1]) - sx(edges[i]) - 2, 1);
      s += `<rect class="c-bar" x="${x.toFixed(1)}" y="${sy(c).toFixed(1)}" width="${w.toFixed(1)}" ` +
           `height="${(H - m.b - sy(c)).toFixed(1)}"><title>${compact(edges[i])} to ${compact(edges[i + 1])}: ${c} houses</title></rect>`;
    });
    niceTicks(min, max, 5).forEach((t) => {
      s += `<text class="c-text" x="${sx(t)}" y="${H - m.b + 16}" text-anchor="middle">${compact(t)}</text>`;
    });
    if (min < 0 && max > 0) s += `<line class="c-line" x1="${sx(0)}" x2="${sx(0)}" y1="${m.t}" y2="${H - m.b}"/>`;
    s += `<line class="c-axis" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
    s += `<text class="c-text" x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xLabel || "")}</text>`;
    put(el, s + "</svg>");
  }

  /* dataset price range with the prediction interval and the predicted value */
  function rangeBar(el, { min, max, lo, hi, value }) {
    const W = 560, H = 64, x0 = 10, x1 = W - 10;
    const sx = (v) => x0 + (Math.min(Math.max(v, min), max) - min) / (max - min) * (x1 - x0);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted price compared with the dataset price range">`;
    s += `<rect class="c-track" x="${x0}" y="29" width="${x1 - x0}" height="6" rx="3"/>`;
    s += `<rect class="c-band" x="${sx(lo)}" y="24" width="${Math.max(sx(hi) - sx(lo), 2)}" height="16" rx="3"/>`;
    s += `<circle class="c-mark" cx="${sx(value)}" cy="32" r="7"/>`;
    s += `<text class="c-text" x="${x0}" y="58">cheapest in dataset: ${compact(min)}</text>`;
    s += `<text class="c-text" x="${x1}" y="58" text-anchor="end">most expensive: ${compact(max)}</text>`;
    put(el, s + "</svg>");
  }

  async function loadReport() {
    const res = await fetch("/api/report");
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not load the report");
    return data;
  }

  return { esc, full, compact, scatter, hbar, histogram, rangeBar, loadReport };
})();
