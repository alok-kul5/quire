(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const TRIES = [
    "Why does the walrus operator need parentheses at the top level?",
    "What did PEP 703 propose for the GIL?",
    "How do I mark positional-only parameters?",
    "Why was a dedicated matrix multiplication operator added?",
  ];
  let mode = "rerank";

  // ---------- load sequence ----------
  setTimeout(() => document.body.classList.add("loaded"), 1900);

  // ---------- health ----------
  fetch("/api/health").then((r) => r.json()).then((h) => {
    const pill = $("#health");
    pill.classList.add(h.llm_available ? "ok" : "warn");
    $("span", pill).textContent = h.llm_available ? `${h.model} ready` : `${h.model} offline`;
    $("#modelName").textContent = h.model;
    $("#chunkCount").textContent = h.chunks.toLocaleString();
  }).catch(() => { $("#health span").textContent = "api offline"; });

  // ---------- controls ----------
  const tries = $("#tries");
  TRIES.forEach((t) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = t.length > 44 ? t.slice(0, 42) + "…" : t;
    b.title = t;
    b.onclick = () => { $("#q").value = t; ask(t); };
    tries.appendChild(b);
  });
  $$(".seg button").forEach((b) => b.addEventListener("click", () => {
    $$(".seg button").forEach((x) => x.classList.toggle("on", x === b));
    mode = b.dataset.mode;
    const q = $("#q").value.trim();
    if (q && !$("#result").hidden) ask(q);
  }));
  $("#askForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("#q").value.trim() || $("#q").placeholder;
    $("#q").value = q;
    ask(q);
  });

  // ---------- pipeline animation ----------
  const STAGES = ["bm25", "dense", "fuse", "rerank", "llm"];
  const ACTIVE = { bm25: ["bm25", "llm"], dense: ["dense", "llm"], hybrid: ["bm25", "dense", "fuse", "llm"], rerank: STAGES };
  function pipe(state, timings = {}) {
    const active = ACTIVE[mode];
    const ms = { bm25: timings.bm25_ms, dense: timings.dense_ms, fuse: null, rerank: timings.rerank_ms, llm: timings.llm_ms };
    STAGES.forEach((s, i) => {
      const li = $(`.pipeline li[data-stage="${s}"]`);
      li.classList.remove("run", "done", "skip");
      if (!active.includes(s)) { li.classList.add("skip"); $(".ms", li).textContent = "off"; return; }
      if (state === "run") { setTimeout(() => li.classList.add("run"), i * 120); $(".ms", li).textContent = "…"; }
      else {
        setTimeout(() => li.classList.add("done"), i * 140);
        const v = ms[s];
        $(".ms", li).textContent = v == null ? (s === "fuse" ? "<1 ms" : "–") : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`;
      }
    });
  }

  // ---------- ask ----------
  let seq = 0;
  async function ask(question) {
    const my = ++seq;
    const res = $("#result");
    res.hidden = false;
    $("#askBtn").disabled = true;
    const ans = $("#answer");
    ans.className = "answer loading";
    ans.innerHTML = '<span class="skeleton" style="width:92%"></span><span class="skeleton" style="width:78%"></span><span class="skeleton" style="width:55%"></span>';
    $("#answerFoot").textContent = "";
    $("#modeTag").textContent = $(`.seg button[data-mode="${mode}"]`).textContent;
    $("#sources").innerHTML = "";
    pipe("run");
    res.scrollIntoView({ behavior: "smooth", block: "start" });
    try {
      const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, mode }) });
      const data = await r.json();
      if (my !== seq) return;
      if (!r.ok) throw new Error(data.detail || r.statusText);
      render(data);
      pipe("done", data.timings);
    } catch (err) {
      if (my !== seq) return;
      ans.className = "answer refused";
      ans.textContent = `Something went wrong: ${err.message}`;
      pipe("done");
    } finally {
      if (my === seq) $("#askBtn").disabled = false;
    }
  }

  function render(d) {
    const ans = $("#answer");
    ans.className = "answer" + (d.refused ? " refused" : "");
    // word-by-word ink-in, citations as live chips
    const parts = esc(d.answer).split(/(\[\d+\])/g);
    let html = "", i = 0;
    for (const p of parts) {
      const m = p.match(/^\[(\d+)\]$/);
      if (m) { html += `<a class="cite w" style="animation-delay:${i * 14}ms" href="#src-${m[1]}" data-n="${m[1]}">${m[1]}</a>`; i++; continue; }
      for (const w of p.split(/(\s+)/)) {
        if (!w) continue;
        html += /^\s+$/.test(w) ? w : `<span class="w" style="animation-delay:${i++ * 14}ms">${w}</span>`;
      }
    }
    ans.innerHTML = html;
    const t = d.timings || {};
    const cited = d.sources.filter((s) => s.cited).map((s) => `PEP ${s.pep}`);
    $("#answerFoot").textContent = d.refused
      ? `Refused: no passage scored high enough to ground an answer · ${d.model}`
      : `${[...new Set(cited)].join(" · ") || "no citations"} · retrieval ${Math.round(t.retrieval_ms || 0)} ms · ${d.model} ${((t.llm_ms || 0) / 1000).toFixed(1)} s`;

    const box = $("#sources");
    box.innerHTML = d.sources.map((s, k) => {
      const rk = s.ranks || {};
      const order = ["bm25", "dense", "fused", "rerank"].filter((x) => rk[x]);
      const final = rk.rerank || rk.fused || rk[order[0]];
      const chips = order.map((x) => {
        const up = x === "rerank" && rk.fused && rk.rerank < rk.fused;
        return `<span class="rk ${up ? "up" : ""}">${x} #${rk[x]}${up ? ` ↑${rk.fused - rk.rerank}` : ""}</span>`;
      }).join("");
      return `<article class="src ${s.cited ? "cited" : ""}" id="src-${s.n}" data-n="${s.n}" style="animation-delay:${120 + k * 70}ms">
        <span class="n">${s.n}</span>
        <div class="src-head"><a href="${esc(s.url)}" target="_blank" rel="noopener">PEP ${s.pep} · ${esc(s.title)}</a><span class="status ${esc(s.status)}">${esc(s.status)}</span></div>
        <div class="src-sec">§ ${esc(s.section)}</div>
        <p>${esc(s.text)}</p>
        <div class="ranks">${chips}<span class="rk">score ${s.score}</span></div>
        <button class="more" type="button">Expand</button>
      </article>`;
    }).join("");
    $$(".src .more", box).forEach((b) => b.onclick = () => { const a = b.closest(".src"); a.classList.toggle("open"); b.textContent = a.classList.contains("open") ? "Collapse" : "Expand"; });
    $$(".cite", ans).forEach((c) => {
      const n = c.dataset.n;
      const on = (v) => { $(`#src-${n}`)?.classList.toggle("hot", v); c.classList.toggle("hot", v); };
      c.addEventListener("mouseenter", () => on(true));
      c.addEventListener("mouseleave", () => on(false));
      c.addEventListener("click", (e) => { e.preventDefault(); const el = $(`#src-${n}`); el?.scrollIntoView({ behavior: "smooth", block: "center" }); on(true); setTimeout(() => on(false), 1200); });
    });
  }

  // ---------- evals ----------
  const NAMES = { bm25: "BM25", dense: "Dense", hybrid: "Hybrid (RRF)", rerank: "Hybrid + Rerank" };
  const pct = (x) => (x == null ? "–" : `${(x * 100).toFixed(1)}%`);
  fetch("/api/evals").then((r) => r.json()).then((e) => {
    const r = e.retrieval;
    if (r) {
      const modes = Object.keys(r);
      const best = modes.reduce((a, b) => (r[b]["mrr@10"] > r[a]["mrr@10"] ? b : a));
      $("#ablation").innerHTML = modes.map((m) => `
        <div class="ab ${m === best ? "best" : ""}">
          <h3>${NAMES[m]}</h3>
          <div class="big">${(r[m]["recall@5"] * 100).toFixed(0)}<small>% recall@5</small></div>
          <div class="bar"><i data-w="${r[m]["mrr@10"] * 100}"></i></div>
          <dl><dt>MRR@10</dt><dd>${r[m]["mrr@10"].toFixed(3)}</dd><dt>Recall@1</dt><dd>${pct(r[m]["recall@1"])}</dd><dt>Recall@10</dt><dd>${pct(r[m]["recall@10"])}</dd><dt>p50</dt><dd>${Math.round(r[m].p50_ms)} ms</dd></dl>
        </div>`).join("");
      const io = new IntersectionObserver((ents) => ents.forEach((en) => {
        if (en.isIntersecting) { $$(".bar i", en.target).forEach((i) => (i.style.width = i.dataset.w + "%")); io.disconnect(); }
      }), { threshold: .3 });
      io.observe($("#ablation"));
    } else {
      $("#ablation").innerHTML = '<p class="sub">Run <code>python -m quire.evaluate retrieval</code> to populate this table.</p>';
    }
    const g = e.generation;
    if (g) {
      $("#gen").innerHTML = [
        [pct(g.judge_faithful), `judged faithful by ${g.judge}`],
        [pct(g.heuristic_supported), "sentences cited and grounded"],
        [pct(g.citation_precision), "citations point at a gold PEP"],
        [pct(g.refusal_rate_unanswerable), "off-topic questions refused"],
        [pct(g.keyword_accuracy), "keyword accuracy"],
      ].map(([v, l]) => `<div class="g"><b>${v}</b><span>${l}</span></div>`).join("");
    }
  }).catch(() => {});
})();
