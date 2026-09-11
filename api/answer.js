// POST /api/answer  { question, sources: [{ id, doc, page, title, text }] }  (sources = top chunks from retrieval)
// The model may only use the numbered sources and must cite them as [n]. Two guards run after the model:
//  1. citations to numbers that were not given are stripped;
//  2. an answer with no valid citation is withheld and replaced by the refusal, so an unsupported answer
//     never reaches the user even if the model ignores the instructions.
import { chat } from "../lib/openrouter.js";

export const REFUSAL = "I can't find that in the documents.";

const SYSTEM = [
  "You answer questions using ONLY the numbered sources provided.",
  "Rules:",
  "1. End every sentence that states a fact with the number of the source it came from, in square brackets, like [2] or [1][3].",
  `2. If the sources do not contain the answer, reply with exactly: ${REFUSAL}`,
  "3. Never use outside knowledge. Never guess prices, dates, names or policies.",
  "4. Be brief: one to four sentences.",
].join("\n");

export function postProcess(text, sources) {
  const t = String(text).trim();
  const refused = /can.?t find that in the documents|not in the documents/i.test(t);
  const cited = [...new Set([...t.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))];
  const valid = cited.filter((n) => n >= 1 && n <= sources.length);
  const invalid = cited.filter((n) => !valid.includes(n));
  const answer = t.replace(/\[(\d+)\]/g, (m, n) => (Number(n) >= 1 && Number(n) <= sources.length ? m : "")).trim();
  if (refused) return { answer: REFUSAL, refused: true, citations: [], invalid_citations: invalid };
  if (!valid.length) return { answer: REFUSAL, refused: true, guard: "The model's answer had no valid citation, so it was withheld.", citations: [], invalid_citations: invalid };
  return { answer, refused: false, citations: valid.map((n) => ({ n, id: sources[n - 1].id })), invalid_citations: invalid };
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const { question, sources } = req.body || {};
  if (typeof question !== "string" || !question.trim() || question.length > 500)
    return res.status(400).json({ error: "question must be 1-500 characters" });
  if (!Array.isArray(sources) || !sources.length || sources.length > 8)
    return res.status(400).json({ error: "sources must be 1-8 retrieved chunks" });

  const srcs = sources.map((s, i) => ({
    n: i + 1,
    id: String(s.id ?? i + 1).slice(0, 120),
    label: `${s.doc ?? ""} ${s.page ?? ""} ${s.title ?? ""}`.trim().slice(0, 200),
    text: String(s.text ?? "").slice(0, 1800),
  }));
  const user = `Sources:\n\n${srcs.map((s) => `[${s.n}] (${s.label})\n${s.text}`).join("\n\n")}\n\nQuestion: ${question.trim()}`;

  const t0 = Date.now();
  try {
    const r = await chat({ messages: [{ role: "system", content: SYSTEM }, { role: "user", content: user }], maxTokens: 400, temperature: 0 });
    return res.status(200).json({ ...postProcess(r.content, srcs), model: r.model, latency_ms: Date.now() - t0, usage: r.usage, tried: r.tried });
  } catch (e) {
    return res.status(502).json({ error: e.message.slice(0, 400), tried: e.tried || [] });
  }
}
