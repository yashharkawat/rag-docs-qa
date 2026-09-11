// POST /api/judge  { question, answer, sources }  -> { faithful, unsupported[] }
// LLM-as-judge for the eval page: is every factual claim in the answer backed by the sources it was given?
// A second model call is cheap here, and it catches the failure citations alone can't: a real [n] on a wrong claim.
import { chat, parseJsonObject } from "../lib/openrouter.js";

const SYSTEM =
  'You check answers for faithfulness. Return ONLY a JSON object: {"faithful": boolean, "unsupported": string[]}. ' +
  '"faithful" is true only if every factual claim in the ANSWER is directly supported by the SOURCES. ' +
  'List any unsupported claims, quoted briefly, in "unsupported". A refusal that says the documents do not contain the answer counts as faithful.';

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const { question, answer, sources } = req.body || {};
  if (typeof answer !== "string" || !Array.isArray(sources)) return res.status(400).json({ error: "answer and sources are required" });
  const src = sources.slice(0, 8).map((s, i) => `[${i + 1}] ${String(s.text ?? "").slice(0, 1800)}`).join("\n\n");
  const user = `SOURCES:\n${src}\n\nQUESTION: ${String(question ?? "").slice(0, 500)}\n\nANSWER: ${answer.slice(0, 2000)}`;
  try {
    const r = await chat({ messages: [{ role: "system", content: SYSTEM }, { role: "user", content: user }], json: true, maxTokens: 300 });
    const { obj, error } = parseJsonObject(r.content);
    if (!obj) return res.status(502).json({ error: `judge reply was not JSON (${error})` });
    return res.status(200).json({
      faithful: obj.faithful === true,
      unsupported: Array.isArray(obj.unsupported) ? obj.unsupported.slice(0, 5).map(String) : [],
      model: r.model,
      latency_ms: r.latency_ms,
    });
  } catch (e) {
    return res.status(502).json({ error: e.message.slice(0, 400) });
  }
}
