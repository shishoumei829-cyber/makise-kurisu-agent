# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Added `README.md` (CN) and `README_EN.md` for product/open-source onboarding
- Added `CONTRIBUTING.md`, `SECURITY.md`, and `RESUME_PITCH.md`
- Added minimal Node test suite (`tests/*.test.js`) for core pure modules
- Added GitHub Actions CI workflow at `.github/workflows/ci.yml`
- Added `release:win` npm script for Windows package build

### Changed

- Updated root `.gitignore` with runtime/cache/binary exclusions
- Updated Electron `main.js` to auto-ensure backend service on startup and stop spawned backend on quit
- Updated `启动实验终端.bat` to use relative path for portability

### Open Source & Distribution

- Added root `README.md`, `LICENSE` (ISC), and `NOTICE.md` (character IP boundary)
- Added `INSTALL.md` and `scripts/install-check.bat` for end users
- Added GitHub Issue/PR templates
- Fixed Electron packaging (`ELECTRON_RUN_AS_NODE`, `signAndEditExecutable: false`)
- Added `scripts/build-release.bat` for reproducible Windows installer builds
- Generated installer: `dist/Amadeus Setup 1.0.0.exe`
- Added frontend startup banner and block chat when system not ready
- Gated debug logging behind `AMADEUS_DEBUG=1`
- Removed hardcoded SoVITS paths from `run_backend.bat` and `env.example`
- Unified backend port via `AMADEUS_BACKEND_PORT`
- Added `MVP_STATUS.md` non-technical product status doc
- Added `lib/startupCheck.js` and tests
