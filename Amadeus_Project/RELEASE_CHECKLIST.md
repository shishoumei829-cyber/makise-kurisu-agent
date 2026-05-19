# Release Checklist (Windows)

## 1. Pre-check

- [ ] `npm install` completed without errors
- [ ] Ollama is reachable at configured `AMADEUS_OLLAMA_BASE`
- [ ] `.env` is configured for target machine (no hardcoded personal paths)
- [ ] RAG index is built (`npm run ingest`) if `brain_data/` changed

## 2. Quality Gate

- [ ] `npm test` passes
- [ ] Manual smoke test:
  - [ ] open app
  - [ ] send one chat message
  - [ ] verify `/pad-state` updates
  - [ ] verify fallback behavior when optional service (TTS) is unavailable

## 3. Build

- [ ] `npm run release:win`
- [ ] Verify generated installer exists in `dist/`
- [ ] Install on a clean Windows machine and run end-to-end smoke test

## 4. Release Artifacts

- [ ] Update `CHANGELOG.md`
- [ ] Tag version in git
- [ ] Upload installer and release notes
- [ ] Include known limitations and optional dependencies in release notes

## 5. Post-release

- [ ] Create issue for regressions discovered during packaging
- [ ] Track startup failures and dependency compatibility feedback
