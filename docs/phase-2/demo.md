# Phase 2 Gradio demo

`app.py` includes a Phase 2 tab with a three-model selector, calibrated probabilities, and a manual-inspection uncertainty banner. The model helper is `src/waste_sorting/phase2_demo.py`. Model temperatures, thresholds, and hashes are loaded from the frozen `docs/phase-2/evaluation/protocol.json`.

## Run

```bash
python -m pip install -e '.[dev,demo,dl]'
python -m pytest -q tests/test_phase2_demo.py
python -m ruff check app.py src/waste_sorting/phase2_demo.py tests/test_phase2_demo.py
python app.py --help
```

The large final model artifacts are not committed. To run real inference, restore the team-supplied A2.1 and H3 model artifacts at the paths expected by `phase2_demo.py`, with exact SHA-256 values in `protocol.json`. Do not disable hash checking to make the demo run.

**Evidence pending:** capture screenshots of H3, A2.1, and SVM predictions plus an uncertainty case after installing the frozen artifacts. Mock-based unit tests do not establish real-model consistency with the frozen validation predictions.
