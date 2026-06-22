/* Setting 3 splits viewer.
 *
 * Loads splits.json, renders one card per split grouped by substrate.
 * Each card has two columns of end-of-training rollout GIFs (re-using
 * the assets from the per-scenario page) with a shift arrow between
 * them. Clicking a thumbnail jumps to the corresponding scenario page.
 */

(async () => {
  const root = document.getElementById("splits-root");

  let doc;
  try {
    const res = await fetch("splits.json", { cache: "no-cache" });
    doc = await res.json();
  } catch (err) {
    root.innerHTML = `<p class="muted">Failed to load splits.json: ${err}</p>`;
    return;
  }

  const html = [];
  for (const sub of doc.substrates) {
    html.push(`<section class="split-substrate" id="${sub.id}">`);
    html.push(`<h2>${escapeHtml(sub.label)}</h2>`);
    html.push(`<div class="split-grid">`);
    for (const split of sub.splits) {
      html.push(renderSplit(sub.id, split));
    }
    html.push(`</div></section>`);
  }
  root.innerHTML = html.join("");

  // Auto-scroll to a hash-targeted split if one was specified.
  if (location.hash) {
    const id = location.hash.replace(/^#\/?/, "");
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ----------------------------------------------------------------
  function renderSplit(subId, split) {
    const resultBlock = split.result ? `
      <div class="split-result">
        <div class="split-side-label">Normalised return — Train ● vs Test ■</div>
        <img src="${escapeHtml(split.result)}"
             alt="${escapeHtml(split.id)} per-algorithm normalised return on train and held-out test scenarios"
             loading="lazy" />
      </div>` : `
      <div class="split-result split-result-missing">
        <p>No final-eval data yet for this split.</p>
      </div>`;

    return `
      <article class="split-card" id="split-${escapeHtml(split.id)}">
        <header class="split-head">
          <span class="split-tag">${escapeHtml(split.id)}</span>
          ${split.summary ? `<span class="split-summary">${escapeHtml(split.summary)}</span>` : ""}
          <span class="split-counts">${split.train.length} → ${split.test.length}</span>
        </header>
        <div class="split-flow">
          <div class="split-side">
            <div class="split-side-label">Train scenarios</div>
            <div class="split-thumbs">${renderThumbs(subId, split.train)}</div>
          </div>
          <div class="split-arrow" aria-hidden="true">
            <span class="split-arrow-glyph">⟶</span>
            <span class="split-arrow-label">distribution shift</span>
          </div>
          <div class="split-side">
            <div class="split-side-label">Held-out test scenarios</div>
            <div class="split-thumbs">${renderThumbs(subId, split.test)}</div>
          </div>
          ${resultBlock}
        </div>
      </article>
    `;
  }

  function renderThumbs(subId, scenarios) {
    return scenarios.map((scen) => `
      <a class="split-thumb" href="index.html#${encodeURIComponent(subId)}/${encodeURIComponent(scen)}"
         title="${escapeHtml(scen)}">
        <img src="assets/${encodeURIComponent(subId)}/${encodeURIComponent(scen)}.late.gif"
             alt="${escapeHtml(scen)} end-of-training rollout"
             loading="lazy" />
        <span class="split-thumb-cap">${escapeHtml(prettyScen(scen))}</span>
      </a>
    `).join("");
  }

  function prettyScen(scen) {
    // commons_harvest__closed_3 → "Closed 3"; coins_4 → "4"
    if (scen.startsWith("commons_harvest__")) {
      const rest = scen.slice("commons_harvest__".length);
      const m = rest.match(/^(closed|open|partnership)_(\d+)$/);
      if (m) return `${capitalise(m[1])} ${m[2]}`;
    }
    const m = scen.match(/_(\d+)$/);
    return m ? m[1] : scen;
  }

  function capitalise(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }
})();
