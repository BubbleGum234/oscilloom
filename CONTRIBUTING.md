# Contributing to Oscilloom

Thanks for your interest in contributing! Oscilloom is a local-first EEG pipeline builder, and we welcome contributions from neuroscience researchers, developers, and anyone who wants to make EEG processing more accessible.

## Getting Started

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/oscilloom.git
cd oscilloom

# Backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

## Development Workflow

1. **Backend**: `uvicorn backend.main:app --reload --port 8000`
2. **Frontend**: `cd frontend && npm run dev`
3. **Tests**: `python -m pytest backend/tests/ -v`

All 741+ tests must pass before submitting a PR.

## Adding a New Node

This is the most common contribution. It requires exactly **2 file changes**:

1. **Define the node** in `backend/registry/nodes/<category>.py`:
   - Write an `execute_fn` following the contract (copy-on-write, `verbose=False`, sync)
   - Create a `NodeDescriptor` with all fields

2. **Register it** in `backend/registry/__init__.py`:
   - Import and add to `NODE_REGISTRY`

The engine, validation, export, and frontend palette pick it up automatically.

### execute_fn Contract

```python
def _execute_my_node(input_data, params: dict):
    data = input_data.copy()          # Always copy first
    data.filter(params["l_freq"], params["h_freq"], verbose=False)  # verbose=False
    return data                        # Return new object
```

### Rules
- Must be synchronous (runs in ThreadPoolExecutor)
- Always `raw.copy()` before in-place operations
- Always `verbose=False` on MNE calls
- Visualization nodes: `plt.close(fig)` after encoding
- Never suppress exceptions

## Code Style

- Python: follow existing patterns in the codebase
- TypeScript: strict mode, no `any` types
- Keep it simple — don't over-engineer

## Testing

- All tests use the synthetic `raw` fixture from `conftest.py`
- Use `TestClient(app)` for route tests
- Name tests: `test_<what>_<expected_outcome>()`
- Keep tests fast — no real files or network calls

## Pull Request Process

1. Create a focused PR (one feature or fix per PR)
2. Include tests for new functionality
3. Update the README if adding user-facing features
4. Ensure CI passes (all tests green, TypeScript compiles)

## Reporting Issues

- Use GitHub Issues
- Include: what you expected, what happened, steps to reproduce
- For EEG processing issues: include the file format and MNE version

## License

By contributing, you agree that your contributions will be licensed under the same [Apache License 2.0](LICENSE.md) as the project.
