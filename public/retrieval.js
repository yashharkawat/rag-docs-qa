// Hybrid retrieval: BM25 keyword scores + cosine similarity of embeddings, fused with Reciprocal Rank Fusion.
// Keywords catch exact terms (prices, "OPG", branch names); embeddings catch paraphrases ("fee" vs "costs").
// Works in the browser and in Node (the smoke test imports it too).

const STOP = new Set(
  "a an the and or of to in on for is are was were be by with at from as it its this that your you we our us can do does did how what when where which who whom will would should could i my me if not no yes there their them then than so".split(" ")
);

export function tokenize(s) {
  return (String(s).toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => !STOP.has(t) && t.length > 1);
}

const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

export class Index {
  constructor(chunks = []) {
    this.chunks = [];
    this.add(chunks);
  }

  add(chunks) {
    this.chunks.push(...chunks);
    this.tf = this.chunks.map((c) => {
      const m = new Map();
      for (const t of tokenize(`${c.title} ${c.text}`)) m.set(t, (m.get(t) || 0) + 1);
      return m;
    });
    this.len = this.tf.map((m) => [...m.values()].reduce((a, b) => a + b, 0));
    this.avgdl = this.len.reduce((a, b) => a + b, 0) / Math.max(1, this.len.length);
    this.df = new Map();
    for (const m of this.tf) for (const t of m.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
  }

  remove(predicate) {
    const keep = this.chunks.filter((c) => !predicate(c));
    this.chunks = [];
    this.add(keep);
  }

  bm25(query, k1 = 1.2, b = 0.75) {
    const q = tokenize(query);
    const N = this.chunks.length;
    return this.tf.map((tf, i) => {
      let s = 0;
      for (const t of q) {
        const f = tf.get(t);
        if (!f) continue;
        const df = this.df.get(t) || 0;
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        s += (idf * (f * (k1 + 1))) / (f + k1 * (1 - b + (b * this.len[i]) / this.avgdl));
      }
      return s;
    });
  }

  /** Returns the top k chunks with their fused score and both raw signals, best first. */
  search(query, queryVector, k = 5) {
    if (!this.chunks.length) return [];
    const bm = this.bm25(query);
    const cos = this.chunks.map((c) => dot(queryVector, c.embedding));
    const ranks = (scores) => {
      const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
      const r = new Array(scores.length);
      order.forEach(([, i], pos) => (r[i] = pos + 1));
      return r;
    };
    const rb = ranks(bm);
    const rv = ranks(cos);
    const K = 60; // standard RRF constant
    return this.chunks
      .map((c, i) => ({ c, i, score: 1 / (K + rv[i]) + (bm[i] > 0 ? 1 / (K + rb[i]) : 0), bm25: bm[i], cos: cos[i] }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(({ c, score, bm25, cos }) => ({ ...c, score: +score.toFixed(5), bm25: +bm25.toFixed(3), cos: +cos.toFixed(3) }));
  }
}
