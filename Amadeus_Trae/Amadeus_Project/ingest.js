const fs = require("fs");
const path = require("path");

const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { OllamaEmbeddings } = require("@langchain/ollama");

let HNSWLib = null;
try {
  ({ HNSWLib } = require("@langchain/community/vectorstores/hnswlib"));
} catch (err) {
  console.warn("[ingest] HNSWLib unavailable, fallback to JSON store:", err.message);
}

const ROOT_DIR = path.resolve(__dirname);
const BRAIN_DIR = path.join(ROOT_DIR, "brain_data");
const VECTOR_DIR = path.join(ROOT_DIR, "vector_store");
const FALLBACK_JSON = path.join(VECTOR_DIR, "store.json");

function walkTxtFiles(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...walkTxtFiles(full));
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".txt")) {
      out.push(full);
    }
  }
  return out;
}

async function build() {
  if (!fs.existsSync(BRAIN_DIR)) {
    throw new Error(`brain_data folder not found: ${BRAIN_DIR}`);
  }

  const txtFiles = walkTxtFiles(BRAIN_DIR);
  if (!txtFiles.length) {
    throw new Error(`No .txt files found in ${BRAIN_DIR}`);
  }

  const rawDocs = txtFiles.map((file) => ({
    pageContent: fs.readFileSync(file, "utf8"),
    metadata: { source: file },
  }));

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 80,
  });
  const chunks = await splitter.splitDocuments(rawDocs);

  if (!chunks.length) {
    throw new Error("No chunks generated from brain_data.");
  }

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://127.0.0.1:11434",
  });

  fs.mkdirSync(VECTOR_DIR, { recursive: true });

  if (HNSWLib) {
    try {
      const store = await HNSWLib.fromDocuments(chunks, embeddings);
      await store.save(VECTOR_DIR);
      console.log(`[ingest] HNSW vector store saved to ${VECTOR_DIR}`);
      console.log(`[ingest] indexed chunks: ${chunks.length}`);
      return;
    } catch (err) {
      console.warn("[ingest] HNSW build failed, fallback to JSON store:", err.message);
    }
  }

  const texts = chunks.map((d) => d.pageContent);
  const vectors = await embeddings.embedDocuments(texts);
  const serializable = chunks.map((chunk, i) => ({
    id: i,
    text: chunk.pageContent,
    metadata: chunk.metadata || {},
    embedding: vectors[i],
  }));
  fs.writeFileSync(FALLBACK_JSON, JSON.stringify(serializable, null, 2), "utf8");
  console.log(`[ingest] JSON vector fallback saved to ${FALLBACK_JSON}`);
  console.log(`[ingest] indexed chunks: ${chunks.length}`);
}

build().catch((err) => {
  console.error("[ingest] failed:", err.message);
  process.exit(1);
});
