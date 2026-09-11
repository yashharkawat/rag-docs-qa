// Chunking, shared by the browser (uploads) and scripts/build-corpus.mjs (sample docs).
// Markdown is split by heading, so a chunk never straddles two topics. Long sections are packed
// paragraph by paragraph up to maxChars, and a paragraph that is still too long is split at sentence ends.

export function chunkMarkdown(md, doc, maxChars = 900) {
  const lines = md.replace(/\r/g, "").split("\n");
  const sections = [];
  let path = [];
  let cur = null;
  for (const line of lines) {
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) {
      const level = h[1].length;
      path = path.slice(0, level - 1);
      path[level - 1] = h[2].trim();
      cur = { title: path.filter(Boolean).join(" › "), body: [] };
      sections.push(cur);
    } else {
      if (!cur) {
        cur = { title: doc, body: [] };
        sections.push(cur);
      }
      cur.body.push(line);
    }
  }
  const chunks = [];
  let sectionNo = 0;
  for (const s of sections) {
    const text = s.body.join("\n").trim();
    if (!text) continue;
    sectionNo++;
    for (const piece of packParagraphs(text, maxChars)) chunks.push({ doc, title: s.title, page: `§${sectionNo}`, text: piece });
  }
  return chunks;
}

/** pages: [{ page: 1, text: "..." }] from a PDF. The citation keeps the real page number. */
export function chunkPages(pages, doc, maxChars = 900) {
  const out = [];
  for (const p of pages) {
    const text = p.text.replace(/[ \t]+/g, " ").trim();
    if (!text) continue;
    for (const piece of packParagraphs(text, maxChars)) out.push({ doc, title: `${doc}, page ${p.page}`, page: `p.${p.page}`, text: piece });
  }
  return out;
}

function packParagraphs(text, max) {
  const paras = text.split(/\n\s*\n/).map((p) => p.replace(/\s*\n\s*/g, " ").trim()).filter(Boolean);
  const out = [];
  let buf = "";
  for (const p of paras) {
    if (p.length > max) {
      if (buf) out.push(buf), (buf = "");
      out.push(...splitLong(p, max));
      continue;
    }
    if (buf && (buf + "\n\n" + p).length > max) {
      out.push(buf);
      buf = p;
    } else buf = buf ? buf + "\n\n" + p : p;
  }
  if (buf) out.push(buf);
  return out;
}

function splitLong(p, max) {
  const sentences = p.match(/[^.!?]+[.!?]+(\s+|$)|[^.!?]+$/g) || [p];
  const out = [];
  let buf = "";
  for (const s of sentences) {
    if (buf && (buf + s).length > max) {
      out.push(buf.trim());
      buf = s;
    } else buf += s;
  }
  if (buf.trim()) out.push(buf.trim());
  // a single sentence longer than max (tables, scanned text) is hard-split
  return out.flatMap((x) => (x.length > max ? x.match(new RegExp(`[\\s\\S]{1,${max}}`, "g")) : [x]));
}
