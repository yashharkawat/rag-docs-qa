// Builds public/corpus/index.json from corpus/*.md: chunk by heading, then embed each chunk with all-MiniLM-L6-v2.
// The browser embeds questions with the same model and dtype (q8), so questions and chunks share one vector space.
import fs from "node:fs";
import path from "node:path";
import { pipeline } from "@huggingface/transformers";
import { chunkMarkdown } from "../public/chunk.js";

const SRC = "corpus";
const OUT = "public/corpus/index.json";
const MODEL = "Xenova/all-MiniLM-L6-v2";

const files = fs.readdirSync(SRC).filter((f) => f.endsWith(".md")).sort();
const chunks = [];
for (const f of files) {
  const doc = f.replace(/\.md$/, "");
  chunkMarkdown(fs.readFileSync(path.join(SRC, f), "utf8"), doc).forEach((c, i) => chunks.push({ id: `${doc}#${i + 1}`, ...c }));
}

const extractor = await pipeline("feature-extraction", MODEL, { dtype: "q8" });
for (let i = 0; i < chunks.length; i += 16) {
  const batch = chunks.slice(i, i + 16);
  const out = await extractor(batch.map((c) => `${c.title}\n${c.text}`), { pooling: "mean", normalize: true });
  out.tolist().forEach((v, j) => (batch[j].embedding = v.map((x) => Math.round(x * 1e4) / 1e4)));
}

fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify({ model: MODEL, dtype: "q8", dims: chunks[0].embedding.length, built: new Date().toISOString(), chunks }));
console.log(`wrote ${chunks.length} chunks from ${files.length} docs -> ${OUT}`);
