import { Index } from "./retrieval.js";
import { chunkMarkdown, chunkPages } from "./chunk.js";

const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";
const PDFJS = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
const PDF_WORKER = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";
const MODEL = "Xenova/all-MiniLM-L6-v2";
const SUGGESTIONS = [
  "How much does a zirconia crown cost?",
  "What happens if I miss an appointment?",
  "Is the clinic open on Sunday?",
  "What should I avoid after teeth whitening?",
  "Do you offer laser gum surgery?",
];

const $ = (s) => document.querySelector(s);
const state = { index: new Index(), docs: new Map(), extractorPromise: null };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const setStatus = (t) => ($("#status").textContent = t);
const strip = ({ embedding, ...c }) => c;

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== `tab-${b.dataset.tab}`));
  })
);

// ---------- embedding model (runs in the browser) ----------
function getExtractor() {
  if (!state.extractorPromise) {
    state.extractorPromise = (async () => {
      setStatus("Loading the embedding model in your browser (about 23 MB, first visit only)…");
      const { pipeline, env } = await import(TRANSFORMERS);
      env.allowLocalModels = false;
      const load = () =>
        pipeline("feature-extraction", MODEL, {
          dtype: "q8",
          progress_callback: (p) => {
            if (p.status === "progress" && String(p.file).endsWith(".onnx")) setStatus(`Downloading the embedding model… ${Math.round(p.progress)}%`);
          },
        });
      // The model files come from the Hugging Face CDN; a single dropped request shouldn't break the page.
      let ex;
      for (let attempt = 1; ; attempt++) {
        try {
          ex = await load();
          break;
        } catch (e) {
          if (attempt >= 3) throw e;
          setStatus(`Model download hiccup (${e.message}), retrying ${attempt + 1}/3…`);
          await new Promise((r) => setTimeout(r, 1500 * attempt));
        }
      }
      setStatus(`Ready. ${state.index.chunks.length} passages from ${state.docs.size} documents. Embeddings run locally in your browser.`);
      return ex;
    })().catch((e) => {
      state.extractorPromise = null;
      setStatus("Could not load the embedding model: " + e.message);
      throw e;
    });
  }
  return state.extractorPromise;
}
async function embed(texts) {
  const ex = await getExtractor();
  return (await ex(texts, { pooling: "mean", normalize: true })).tolist();
}

// ---------- documents ----------
async function loadSamples() {
  const corpus = await (await fetch("/corpus/index.json")).json();
  state.index.add(corpus.chunks);
  for (const c of corpus.chunks) {
    const d = state.docs.get(c.doc) || { source: "sample", chunks: 0 };
    d.chunks++;
    state.docs.set(c.doc, d);
  }
  renderDocs();
  setStatus(`${corpus.chunks.length} passages from ${state.docs.size} sample documents (a fictional dental clinic's handbook). Loading the embedding model…`);
  getExtractor().catch(() => {});
}

function renderDocs() {
  $("#doc-rows").innerHTML = [...state.docs]
    .map(([doc, d]) => `<tr><td>${esc(doc)}</td><td>${d.source}</td><td>${d.chunks}</td></tr>`)
    .join("") || `<tr><td colspan="3" class="na">No documents loaded. Upload one above.</td></tr>`;
}

async function extractPdf(file) {
  const pdfjs = await import(PDFJS);
  pdfjs.GlobalWorkerOptions.workerSrc = PDF_WORKER;
  const pdf = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
  const pages = [];
  for (let i = 1; i <= Math.min(pdf.numPages, 200); i++) {
    const tc = await (await pdf.getPage(i)).getTextContent();
    pages.push({ page: i, text: tc.items.map((it) => it.str + (it.hasEOL ? "\n" : " ")).join("") });
  }
  return pages;
}

async function handleFiles(files) {
  for (const file of files) {
    if (file.size > 10 * 1024 * 1024) { setStatus(`${file.name} is over 10 MB, skipped.`); continue; }
    const doc = file.name.replace(/\.[^.]+$/, "").slice(0, 60);
    setStatus(`Reading ${file.name}…`);
    let chunks;
    try {
      chunks = /\.pdf$/i.test(file.name) ? chunkPages(await extractPdf(file), doc) : chunkMarkdown(await file.text(), doc);
    } catch (e) {
      setStatus(`Could not read ${file.name}: ${e.message}`);
      continue;
    }
    if (!chunks.length) { setStatus(`${file.name} has no extractable text (a scanned PDF needs OCR first).`); continue; }
    chunks = chunks.slice(0, 400);
    chunks.forEach((c, i) => (c.id = `${doc}#${i + 1}`));
    for (let i = 0; i < chunks.length; i += 16) {
      const batch = chunks.slice(i, i + 16);
      (await embed(batch.map((c) => `${c.title}\n${c.text}`))).forEach((v, j) => (batch[j].embedding = v));
      setStatus(`Embedding ${file.name}: ${Math.min(i + 16, chunks.length)} / ${chunks.length} passages`);
    }
    state.index.remove((c) => c.doc === doc);
    state.index.add(chunks);
    state.docs.set(doc, { source: "uploaded", chunks: chunks.length });
    renderDocs();
    setStatus(`Added ${file.name}: ${chunks.length} passages. ${state.index.chunks.length} passages loaded in total.`);
  }
}
$("#file").addEventListener("change", (e) => handleFiles([...e.target.files]).finally(() => (e.target.value = "")));
$("#clear-samples").addEventListener("click", () => {
  state.index.remove((c) => state.docs.get(c.doc)?.source === "sample");
  for (const [doc, d] of state.docs) if (d.source === "sample") state.docs.delete(doc);
  renderDocs();
  setStatus("Sample documents removed. Upload your own to ask about them (the Eval tab needs the samples).");
});

// ---------- ask ----------
async function retrieve(q) {
  const t0 = performance.now();
  const [qv] = await embed([q]);
  const top = state.index.search(q, qv, 5);
  return { top, ms: Math.round(performance.now() - t0) };
}
async function callAnswer(question, top) {
  const r = await fetch("/api/answer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, sources: top.map(strip) }) });
  return r.json();
}

function renderAnswer(j, top) {
  const box = $("#answer");
  box.hidden = false;
  box.classList.toggle("refused", !!j.refused || !!j.error);
  if (j.error) { box.innerHTML = `The model call failed: ${esc(j.error)}`; return; }
  box.innerHTML =
    esc(j.answer).replace(/\[(\d+)\]/g, (m, n) => `<a class="cite" data-n="${n}" title="${esc(top[n - 1]?.title || "")}">${n}</a>`) +
    (j.guard ? `<span class="guard">${esc(j.guard)}</span>` : "");
  box.querySelectorAll(".cite").forEach((a) =>
    a.addEventListener("click", () => {
      const li = document.querySelector(`#sources li[data-n="${a.dataset.n}"]`);
      if (li) { li.scrollIntoView({ behavior: "smooth", block: "center" }); li.classList.remove("flash"); void li.offsetWidth; li.classList.add("flash"); }
    })
  );
}

function renderSources(top, citedNs = []) {
  $("#src-h").hidden = !top.length;
  $("#sources").innerHTML = top
    .map((c, i) => `<li data-n="${i + 1}" class="${citedNs.includes(i + 1) ? "cited" : ""}">
      <div class="src-head"><span><b>[${i + 1}] ${esc(c.doc)}</b> · ${esc(c.page)} · ${esc(c.title)}</span>
      <span class="scores">rrf ${c.score} · bm25 ${c.bm25} · cos ${c.cos}</span></div>
      <div class="src-text">${esc(c.text)}</div></li>`)
    .join("");
}

async function ask(q) {
  q = q.trim();
  if (!q) return;
  if (!state.index.chunks.length) { setStatus("No documents loaded. Add one on the Documents tab."); return; }
  $("#ask-btn").disabled = true;
  $("#answer").hidden = true;
  $("#meta").textContent = "Retrieving…";
  try {
    const { top, ms } = await retrieve(q);
    renderSources(top);
    $("#meta").textContent = `Retrieved 5 passages in ${ms} ms. Asking the model…`;
    const j = await callAnswer(q, top);
    renderAnswer(j, top);
    renderSources(top, (j.citations || []).map((c) => c.n));
    $("#meta").textContent = j.error
      ? ""
      : `retrieval ${ms} ms · answer ${j.latency_ms} ms · model ${j.model} · tokens ${j.usage?.prompt_tokens ?? "?"} in / ${j.usage?.completion_tokens ?? "?"} out · cost $0 (free model)`;
  } catch (e) {
    $("#meta").textContent = "Error: " + e.message;
  } finally {
    $("#ask-btn").disabled = false;
  }
}
$("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("#q").value); });
$("#suggest").innerHTML = SUGGESTIONS.map((s) => `<button type="button">${esc(s)}</button>`).join("");
$("#suggest").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { $("#q").value = b.textContent; ask(b.textContent); }));

// ---------- eval ----------
const mark = (v) => (v === null || v === undefined ? `<span class="na">n/a</span>` : v ? `<span class="ok">yes</span>` : `<span class="no">no</span>`);
async function runEval() {
  if (![...state.docs.values()].some((d) => d.source === "sample")) { setStatus("The eval runs on the sample documents. Reload the page to restore them."); return; }
  const btn = $("#run-eval");
  btn.disabled = true;
  const set = await (await fetch("/eval-set.json")).json();
  const rows = [];
  for (const [i, item] of set.entries()) {
    btn.textContent = `Running ${i + 1} / ${set.length}…`;
    const isGold = (c) => c && item.gold.some((g) => c.doc === g.doc && c.title.toLowerCase().includes(g.title.toLowerCase()));
    const answerable = item.gold.length > 0;
    const { top } = await retrieve(item.q);
    const a = await callAnswer(item.q, top);
    const row = { q: item.q, answerable, hit: answerable ? top.some(isGold) : null, model: a.model || "error" };
    row.refusalOk = a.error ? null : answerable ? !a.refused : !!a.refused;
    row.citedGold = answerable && !a.refused && !a.error ? (a.citations || []).some((c) => isGold(top[c.n - 1])) : null;
    row.faithful = null;
    if (answerable && !a.refused && !a.error) {
      const jr = await (await fetch("/api/judge", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: item.q, answer: a.answer, sources: top.map(strip) }) })).json();
      row.faithful = jr.error ? null : jr.faithful;
    }
    rows.push(row);
    $("#eval-rows").innerHTML = rows
      .map((r, k) => `<tr><td>${k + 1}</td><td>${esc(r.q)}${r.answerable ? "" : " <small class='na'>(not in docs)</small>"}</td><td>${mark(r.hit)}</td><td>${mark(r.citedGold)}</td><td>${mark(r.refusalOk)}</td><td>${mark(r.faithful)}</td><td><small>${esc(r.model)}</small></td></tr>`)
      .join("");
  }
  const pct = (xs) => { const v = xs.filter((x) => x !== null); return v.length ? `${Math.round((100 * v.filter(Boolean).length) / v.length)}%` : "n/a"; };
  const ans = rows.filter((r) => r.answerable);
  $("#eval-summary").innerHTML = [
    ["Retrieval hit@5", pct(ans.map((r) => r.hit))],
    ["Cited the right passage", pct(ans.map((r) => r.citedGold ?? false))],
    ["Refusal behaviour correct", pct(rows.map((r) => r.refusalOk))],
    ["Faithful (judged)", pct(ans.map((r) => r.faithful))],
  ].map(([k, v]) => `<div><b>${v}</b>${k}</div>`).join("");
  btn.textContent = "Run the eval again";
  btn.disabled = false;
}
$("#run-eval").addEventListener("click", () => runEval().catch((e) => { setStatus("Eval error: " + e.message); $("#run-eval").disabled = false; }));

loadSamples().catch((e) => setStatus("Could not load the sample documents: " + e.message));
