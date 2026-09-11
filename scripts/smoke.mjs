// Smoke test, run before every push: retrieval hit-rate on the eval set, then one live answerable question
// and one unanswerable one against the real answer handler. Needs OPENROUTER_API_KEY. Exits 1 on failure.
import fs from "node:fs";
import { pipeline } from "@huggingface/transformers";
import { Index } from "../public/retrieval.js";
import answer from "../api/answer.js";

const corpus = JSON.parse(fs.readFileSync("public/corpus/index.json", "utf8"));
const evalSet = JSON.parse(fs.readFileSync("public/eval-set.json", "utf8"));
const idx = new Index(corpus.chunks);
const extractor = await pipeline("feature-extraction", corpus.model, { dtype: corpus.dtype });
const embed = async (q) => (await extractor([q], { pooling: "mean", normalize: true })).tolist()[0];
const isGold = (c, gold) => gold.some((g) => c.doc === g.doc && c.title.toLowerCase().includes(g.title.toLowerCase()));

let hits = 0;
let answerable = 0;
const misses = [];
for (const item of evalSet) {
  if (!item.gold.length) continue;
  answerable++;
  const top = idx.search(item.q, await embed(item.q), 5);
  if (top.some((c) => isGold(c, item.gold))) hits++;
  else misses.push(item.q);
}
console.log(`retrieval hit@5: ${hits}/${answerable}${misses.length ? "  misses: " + misses.join(" | ") : ""}`);

const call = async (question) => {
  const sources = idx.search(question, await embed(question), 5);
  let status = 0;
  let body;
  const res = { status(c) { status = c; return this; }, json(o) { body = o; return this; }, setHeader() {} };
  await answer({ method: "POST", body: { question, sources } }, res);
  return { status, body };
};
const a1 = await call("How much is a root canal for a molar?");
console.log("answerable  :", a1.status, JSON.stringify(a1.body).slice(0, 260));
const a2 = await call("Do you offer laser gum surgery?");
console.log("unanswerable:", a2.status, JSON.stringify(a2.body).slice(0, 260));

const ok =
  hits >= answerable - 1 &&
  a1.status === 200 && !a1.body.refused && a1.body.citations.length > 0 && /8,?500/.test(a1.body.answer) &&
  a2.status === 200 && a2.body.refused === true;
console.log(ok ? "SMOKE PASS" : "SMOKE FAIL");
process.exitCode = ok ? 0 : 1; // not process.exit(): onnxruntime can crash on a forced exit
