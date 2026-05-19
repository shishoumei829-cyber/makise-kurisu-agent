# Amadeus - Kurisu AI Companion

Amadeus is a local-first AI companion project. Instead of a plain chat wrapper, it adds an orchestration layer on top of LLMs: affect (PAD), motivation, strategy, memory, and role-consistency controls.

> Character setting is a fan-work inspired by Steins;Gate and is intended for learning and technical showcase only.

## Highlights

- Local-first runtime with Ollama
- Cognitive pipeline: PAD + goals + strategy + behavior scoring
- Long-term memory + RAG retrieval
- OOC guardrails and response repair
- Optional multimodal modules (vision, TTS, design workflow)
- Optional desktop shell via Electron

## Stack

- Backend: Node.js, Express
- Desktop: Electron
- LLM: Ollama
- Retrieval: LangChain + hnswlib
- Frontend: single-page HTML/CSS/JS UI

## Quick Start

1. Install Node.js (18+)
2. Start Ollama
3. Install dependencies:

```bash
npm install
```

4. Copy `env.example` to `.env` and adjust local values
5. Start backend:

```bash
npm run dev
```

Open:

- `http://localhost:3000`

## Electron

```bash
npm start
```

## Build RAG Index

```bash
npm run ingest
```

## Key Docs

- `PROJECT_STATUS.md`
- `digital_life_upgrade.md`
- `docs/GPU_OLLAMA_SOVITS.md`
- `SECURITY.md`
- `CONTRIBUTING.md`

## Productization Focus

To reach production quality, prioritize:

1. Automated tests (unit + integration)
2. CI pipeline
3. Unified app startup flow
4. Release process and versioning
5. Privacy/compliance documentation
