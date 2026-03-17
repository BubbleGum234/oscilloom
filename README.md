<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0--beta-blue" alt="Version" />
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/react-19-61DAFB?logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/MNE--Python-1.11-green" alt="MNE" />
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License" />
</p>

# Oscilloom

**Visual EEG pipeline builder. Drag. Connect. Discover.**

Process brain recordings without writing a single line of code. Everything runs locally — your data never leaves your machine.

<!-- TODO: Replace with actual demo GIF -->
<!-- ![Oscilloom Demo](docs/assets/demo.gif) -->

---

## Why Oscilloom?

| | MNE-Python | EEGLAB | Brainstorm | **Oscilloom** |
|---|:---:|:---:|:---:|:---:|
| No coding required | | | Partial | **Yes** |
| Free & open | Yes | Needs MATLAB | Yes | **Yes** |
| Runs locally (no cloud) | Yes | Yes | Yes | **Yes** |
| Visual pipeline builder | | | | **Yes** |
| Batch processing | Manual | Manual | Manual | **Built-in** |
| Python script export | N/A | N/A | N/A | **Yes** |
| Modern UI | | | | **Yes** |

## Features

- **40+ processing nodes** — loaders, filters, ICA, PSD, ERPs, connectivity, sleep staging, BCI, and more
- **Drag-and-drop canvas** — build pipelines visually with React Flow
- **Batch processing** — run pipelines across multiple files, export aggregated CSV
- **Compound nodes** — collapse sub-graphs into reusable blocks
- **Data export** — CSV, NPZ, MAT, JSON, FIF, PNG per node
- **Interactive browser** — MNE's data browser with annotation support
- **Script export** — generate standalone Python scripts for reproducibility
- **5 file formats** — EDF, FIF, BrainVision, BDF, CNT

## Quick Start

```bash
# 1. Clone and set up backend
git clone https://github.com/YOUR_USERNAME/oscilloom.git
cd oscilloom
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Start the backend
uvicorn backend.main:app --port 8000 &

# 3. Set up and start the frontend
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** and start building pipelines.

> **Desktop installer coming soon** — no terminal needed.

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Load EDF   │────>│  Bandpass    │────>│ Compute PSD │────>│  Plot PSD   │
│             │     │  1-40 Hz     │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
     I/O            Preprocessing         Analysis          Visualization
```

1. **Upload** an EEG file (EDF, FIF, BrainVision, BDF, or CNT)
2. **Drag** processing nodes from the palette onto the canvas
3. **Connect** them to define your pipeline
4. **Adjust** parameters in the right panel
5. **Run** — results appear on each node, plots render inline

## Node Categories

| Category | Nodes | Examples |
|----------|-------|---------|
| **I/O** | 5 | EDF Loader, FIF Loader, BrainVision Loader, Save to FIF |
| **Preprocessing** | 10+ | Bandpass, Notch, Resample, ICA, Re-reference, Interpolate |
| **Epoching** | 3 | Fixed Epochs, Event Epochs, Epoch Rejection |
| **Analysis** | 5+ | PSD, Band Power, Peak Frequency, Statistics |
| **ERP** | 4 | Average ERP, Grand Average, Peak Latency, Compare Evokeds |
| **Connectivity** | 3 | Spectral Connectivity, Phase-Locking Value |
| **Sleep** | 1 | Automated Sleep Staging |
| **BCI** | 2 | CSP Features, Bandpower Features |
| **Visualization** | 5+ | Plot Raw, Plot PSD, Topomap, ERP Plot, Comparison |

## Architecture

```
Browser (React 19 + TypeScript + React Flow)
    │
    ▼ HTTP/JSON
FastAPI backend (Python 3.11)
    │
    ├── Session Store (in-memory Raw objects)
    ├── DAG Engine (topological sort + execute)
    ├── Node Registry (40+ NodeDescriptors)
    └── ~/.oscilloom/ (persistent storage)
```

- **Generic engine** — never checks node types by name; all logic lives in `execute_fn`
- **Copy-on-write** — stored EEG data is never mutated; every run gets a fresh copy
- **741 tests** — comprehensive backend test coverage

## Supported File Formats

| Format | Extension | Systems |
|--------|-----------|---------|
| EDF/EDF+ | `.edf` | Most clinical EEG systems |
| FIF | `.fif`, `.fif.gz` | MNE native, Elekta/MEGIN MEG |
| BrainVision | `.vhdr` | Brain Products |
| BDF | `.bdf` | BioSemi ActiveTwo |
| CNT | `.cnt` | Neuroscan, ANT Neuro |

## API

The backend exposes a REST API. Once running, explore it at **http://localhost:8000/docs**.

Key endpoints:
- `POST /session/load` — Upload EEG file
- `GET /registry/nodes` — List all available nodes
- `POST /pipeline/execute` — Run a pipeline
- `POST /pipeline/export` — Generate Python script
- `POST /pipeline/report` — Generate PDF report

## Running Tests

```bash
source .venv/bin/activate
python -m pytest backend/tests/ -v    # 741 tests
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI 0.129 + MNE-Python 1.11 + Pydantic v2 |
| Frontend | React 19 + TypeScript 5.9 + React Flow 12 + Tailwind CSS 4 |
| Build | Vite 7 |

## Contributing

We welcome contributions! Whether it's a bug fix, new node type, or documentation improvement.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-node`)
3. Add tests for new functionality
4. Ensure all 741+ tests pass (`python -m pytest backend/tests/ -v`)
5. Submit a pull request

### Adding a new node

Adding a node requires exactly **2 file changes**:
1. Define a `NodeDescriptor` in `backend/registry/nodes/<category>.py`
2. Import and add it to `NODE_REGISTRY` in `backend/registry/__init__.py`

The engine, validation, export, and frontend palette pick it up automatically.

## Roadmap

- [ ] Desktop installer (Tauri — macOS, Windows, Linux)
- [ ] Plugin system for community nodes
- [ ] AI-assisted pipeline builder
- [ ] Real-time streaming support
- [ ] BIDS dataset integration

## License

[Apache License 2.0](LICENSE.md)

---

<p align="center">
  <b>Built for researchers who'd rather study brains than debug code.</b>
</p>
