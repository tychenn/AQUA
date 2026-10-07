# Contributing to AQUA

Use [GitHub Issues](https://github.com/tychenn/AQUA/issues) for bug reports and feature discussions. For a bug report, include the command, traceback, Python version, and relevant package versions.

## CPU tests

Create a dedicated test environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```

The suite covers query budgets, response statistics, watermark path matching, FAISS index outputs, and pipeline branches using temporary data. `tests/test_model_compatibility.py` additionally exercises FAISS and a small Qwen model when the runtime dependencies are installed.

## Changes

Keep changes focused on one behavior, update the corresponding usage instructions, and add a regression test for a corrected bug. Run `python -m pytest -q tests` and `git diff --check` before opening a pull request. Describe the behavior changed and the validation performed.

## Code locations

| Area | Files |
| --- | --- |
| Retrieval and generation | `multimodalrag.py` |
| Watermark paths and query records | `experiments/retrieval_data.py` |
| Index construction and copying | `utils/indexing_faiss.py`, `utils/index_metadata.py` |
| Evaluation | `experiments/{effectiveness,harmlessness,robustness,stealthiness}/` |
| Data layout | `docs/DATA.md` |
