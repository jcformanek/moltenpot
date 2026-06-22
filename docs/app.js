/* Molten Pot — scenario browser SPA.
 *
 * Loads scenarios.json, renders a substrate→scenario tree in the
 * sidebar, and swaps the four asset images in when a scenario is
 * selected. Permalinks live in the URL hash (#substrate/scenario).
 */

(async () => {
  const app = document.getElementById("app");
  const nav = document.getElementById("nav");
  const empty = document.getElementById("viewer-empty");
  const content = document.getElementById("viewer-content");
  const scenTitle = document.getElementById("scen-title");
  const scenId = document.getElementById("scen-id");
  const imgEarly = document.getElementById("scen-early");
  const imgLate  = document.getElementById("scen-late");
  const imgHist  = document.getElementById("scen-hist");
  const imgCmp   = document.getElementById("scen-cmp");

  let manifest;
  try {
    const res = await fetch("scenarios.json", { cache: "no-cache" });
    manifest = await res.json();
  } catch (err) {
    nav.innerHTML = `<p class="muted" style="padding:12px">Failed to load scenarios.json: ${err}</p>`;
    return;
  }

  // Index scenarios by id for O(1) lookup.
  const byId = Object.fromEntries(manifest.scenarios.map((s) => [s.id, s]));

  // Build sidebar tree.
  for (const sub of manifest.substrates) {
    const det = document.createElement("details");
    det.dataset.substrate = sub.id;
    const sum = document.createElement("summary");
    sum.textContent = sub.label;
    det.appendChild(sum);

    const ul = document.createElement("ul");
    for (const scenId of sub.scenarios) {
      const entry = byId[scenId];
      if (!entry) continue;
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = `#${sub.id}/${scenId}`;
      a.dataset.scenId = scenId;
      a.innerHTML = `${entry.label} <code>${escapeHtml(scenId)}</code>`;
      li.appendChild(a);
      ul.appendChild(li);
    }
    det.appendChild(ul);
    nav.appendChild(det);
  }

  // ---------------------------------------------------------------
  // Routing
  // ---------------------------------------------------------------
  function parseHash() {
    const raw = location.hash.replace(/^#\/?/, "");
    if (!raw) return null;
    const [substrate, scenId] = raw.split("/", 2);
    return scenId ? { substrate, scenId } : null;
  }

  function navigateTo(scenId) {
    const entry = byId[scenId];
    if (!entry) {
      empty.hidden = false;
      content.hidden = true;
      empty.querySelector("p").textContent = `Unknown scenario: ${scenId}`;
      return;
    }
    empty.hidden = true;
    content.hidden = false;
    scenTitle.textContent = `${substrateLabel(entry.substrate)} — ${entry.label}`;
    scenId && (document.getElementById("scen-id").textContent = entry.id);

    setImg(imgEarly, entry.early);
    setImg(imgLate,  entry.late);
    setImg(imgHist,  entry.histogram);
    setImg(imgCmp,   entry.comparison);

    // Highlight in sidebar.
    nav.querySelectorAll("a.active").forEach((a) => a.classList.remove("active"));
    const link = nav.querySelector(`a[data-scen-id="${cssEscape(scenId)}"]`);
    if (link) {
      link.classList.add("active");
      const det = link.closest("details");
      if (det) det.open = true;
    }
  }

  function setImg(el, src) {
    if (src) {
      el.src = src;
      el.style.display = "";
    } else {
      el.removeAttribute("src");
      el.style.display = "none";
    }
  }

  function substrateLabel(id) {
    const sub = manifest.substrates.find((s) => s.id === id);
    return sub ? sub.label : id;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function cssEscape(s) {
    return String(s).replace(/(["\\])/g, "\\$1");
  }

  // ---------------------------------------------------------------
  // Wire events.
  // ---------------------------------------------------------------
  window.addEventListener("hashchange", () => {
    const route = parseHash();
    if (route) navigateTo(route.scenId);
  });

  const initial = parseHash();
  if (initial) {
    navigateTo(initial.scenId);
  } else {
    // Default: open the first substrate in the sidebar but do not
    // auto-select a scenario; let the user pick.
    const first = nav.querySelector("details");
    if (first) first.open = true;
  }
  app.dataset.state = "ready";
})();
