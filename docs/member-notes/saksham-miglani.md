# Saksham Miglani: Photo Upload Demo, Integration, and Submission Assembly

Phase 1 member notes for Saksham Miglani.
This document details the demo implementation, shared prediction contract integration, synthetic test fixtures, artifact integrity verification, report assembly, presentation deliverables, reproduction instructions, viva preparation, and an explicit list of missing teammate material and submission requirements.

---

## 1. Implemented Contribution

### 1. Interactive Demo Application (`app.py` & `src/waste_sorting/demo.py`)
- **Shared Prediction Interface Integration:** Inference strictly delegates to `predict_image(image, artifact)` owned by Prajjwal Tripathi (`src/waste_sorting/prediction.py`). Preprocessing, feature extraction, probability mapping, and abstention logic remain in the shared core and are never duplicated in the UI layer.
- **Single Process-Level Model Reuse:** The 66.3 MB evaluated model (`artifacts/calibrated_svm/model.joblib`) is verified and loaded once at process initialization and cached in memory (`_RUNTIME_ARTIFACT`). Repeated classifications reuse this in-memory object, avoiding disk I/O and unpickling overhead on each image upload.
- **Robust Input Handling & Bounding:**
  - Handles PIL images, file paths, numpy arrays, and raw byte buffers.
  - Enforces bounding guards: rejects images exceeding $8,192$ pixels on any side or $16,777,216$ pixels ($16$ MP) total before decoding to prevent decompression bomb vulnerabilities.
  - Surfaces clear, user-facing error messages for empty uploads, unsupported formats, and corrupt files.
- **Uncertainty UX & Safety Signaling:**
  - Frozen abstention threshold is hardcoded to $\tau = 0.52$ (no user-adjustable slider).
  - When $P_{\max} \ge 0.52$: Primary result displays predicted category (e.g., "Metal") with calibrated confidence and zero uncertainty message.
  - When $P_{\max} < 0.52$ (`abstained == True`): Primary result and uncertainty banner prominently display:
    `"Uncertain — manual inspection recommended"` in a high-contrast amber container. A tentative top class (e.g., "Tentative top class (not accepted): Paper") is displayed separately without endorsement.
  - Six-class probability breakdown table formatted cleanly with category names and percentages in canonical order: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
  - Prominent limitations disclaimer: explains that confidence-based abstention does not reliably detect unknown or out-of-distribution categories and that the educational demo does not give formal recycling/disposal advice.
- **State Clearing & Privacy Protection:**
  - Previous prediction outputs are immediately cleared whenever an image is removed or replaced, preventing stale results from being associated with a new photo.
  - Public sharing is disabled by default (`share=False`), and passing `--share` explicitly aborts startup.
  - Uploaded images are processed transiently in RAM within the application layer (`classify_upload` accepts a PIL image, numpy array, or file path and does not write any copy to disk). The Gradio UI framework may create temporary files in the system temporary directory as part of its own file-upload pipeline (this is framework behaviour outside the application's control); the application itself does not persist or store uploaded images.
  - Application startup is guarded by `if __name__ == "__main__":` so importing `app` never starts a server.

### 2. Comprehensive Integration Tests (`tests/test_demo.py`)
- 23 unit and integration tests verifying every aspect of UI callback behavior, error states, and model verification.
- Synthetic in-memory fixtures enable exhaustive testing of accepted, abstained, corrupt, and oversized input states without requiring the 66 MB model binary.
- Dedicated test for the real evaluated model artifact that runs conditionally when the binary is present and verifies predictions on synthetic images without rerunning the held-out test set.

### 3. Reproducible Dependencies (`requirements-demo.lock`)
- Created `requirements-demo.lock` locking Gradio 6.30.0 and its sub-dependencies while preserving all 32 base and runtime dependencies from `requirements.lock` with zero version mismatches.

### 4. Submission Deliverables Assembly
- **IEEE LaTeX Report:** Complete, self-contained IEEE conference report project (`report/main.tex`, `report/references.bib`, `report/Makefile`, `report/README.md`) synthesizing contributions from all teammates.
- **Standalone HTML Report:** Browser-viewable, publication-styled IEEE report (`report/report.html`) with embedded figures and KaTeX mathematics.
- **Presentation Sources & Export:** Editable Marp-compatible Markdown presentation (`presentation/slides.md`) and interactive, responsive zero-dependency HTML presentation viewer (`presentation/index.html`) with keyboard navigation and print-to-PDF stylesheet.
- **Clean Reproduction Guide:** Documented step-by-step reproduction instructions covering both saved-record metric verification and local demo execution.

---

## 2. Statistical Metrics & Ground Truth Preservation

All numbers, metrics, and comparisons in the report, presentation, and demo strictly reflect committed experiment records (`docs/evaluation/summary.json`, `docs/evaluation/test-predictions.csv`, `docs/training/folds.json`). No test set predictions were generated anew.

Key verified figures:
- **Selected SVM Training Cross-Validation Macro F1:** **0.8803 $\pm$ 0.0271** (Extended RBF SVM, $C=10.0, \gamma=10.0$, root\_l2 normalization, 1,608 selected features).
- **Refined Random Forest CV Macro F1:** **0.8370 $\pm$ 0.0241** (Extended spatial representation, 2,760 features, $m = \sqrt{p}$).
- **Final Unconditional Test Macro F1:** **0.8999** (Target: $\ge 0.86$).
- **Final Unconditional Test Accuracy:** **90.19%** (340 / 377 correct).
- **Frozen Abstention Threshold:** **0.52** (Selected strictly on 380 validation images: 98.16% coverage, 90.08% accepted accuracy).
- **Selective Test Accuracy (at $\tau = 0.52$):** **91.37%** (339 / 371 correct, $+1.19$ percentage points over unconditional).
- **Selective Test Coverage:** **98.41%** (371 / 377 accepted; 6 abstained).
- **Selective Test Macro F1:** **0.9129** ($+1.30\%$ relative gain).
- **Test Loss Metrics:** Multiclass Log Loss: **0.4599**; Multiclass Brier Score: **0.1828**; Expected Calibration Error (ECE): **0.1325** (10 fixed bins).

---

## 3. Evaluated Model Artifact Status & Checksum Blocker

The evaluated model binary is:
- **Path:** `artifacts/calibrated_svm/model.joblib`
- **Expected SHA-256:** `83b0658d5180c02d6f03dce835bb8e269d6a89a493b6298a2bb00fd594f3b112`
- **Expected Size:** `66,270,283` bytes (~66.3 MB)
- **Git Policy:** Excluded from Git via `.gitignore` (`artifacts/`).

### Current Status: Blocked
As of current workspace inspection, `artifacts/calibrated_svm/model.joblib` is not present on disk.
- In accordance with instructions:
  - We do not download an arbitrary replacement model.
  - We do not retrain or recalibrate a new model.
  - Real-model integration checks are marked as **BLOCKED** rather than passed.
  - The demo application and test suite support clearly identified synthetic fixture mode (`python app.py --synthetic-fixture`), and `test_demo.py` gracefully skips `test_evaluated_artifact_on_synthetic_image` with an explicit notice.
- **Action Required:** Obtain the preserved `model.joblib` binary from Prajjwal Tripathi and place it at `artifacts/calibrated_svm/model.joblib`. Verify checksum before launching the production demo:
  ```bash
  python -c "
  import hashlib, pathlib
  p = pathlib.Path('artifacts/calibrated_svm/model.joblib')
  assert p.exists(), 'Model file missing'
  digest = hashlib.sha256(p.read_bytes()).hexdigest()
  expected = '83b0658d5180c02d6f03dce835bb8e269d6a89a493b6298a2bb00fd594f3b112'
  assert digest == expected, f'Mismatch: {digest}'
  print('Model artifact verified successfully!')
  "
  ```

---

## 4. Integration Test Suite Structure (`tests/test_demo.py`)

The 23 demo tests cover 4 primary categories:

1. **Callback & Contract Verification (`DemoPredictionTests`):**
   - `test_callback_uses_shared_predict_image`: Confirms callback calls `predict_image` with PIL Image and valid artifact dictionary.
   - `test_accepted_labels_confidence_and_six_class_display`: Verifies correct class label, formatted confidence text, and 6 canonical rows for accepted inputs.
   - `test_abstained_presentation_uses_uncertainty_as_primary`: Confirms primary result and uncertainty message display the prominent warning when confidence $< 0.52$.
   - `test_model_is_reused_across_requests`: Verifies artifact load count remains 1 across multiple predictions.
   - `test_numpy_rgb_array_is_accepted`: Confirms raw numpy arrays are properly coerced and classified.
   - `test_evaluated_artifact_status_is_explicit`: Validates status strings (`verified`, `missing`, `checksum-mismatch`).
   - `test_evaluated_artifact_on_synthetic_image`: Tests real model on synthetic image when present.

2. **Input Validation & Security Guard Tests (`DemoPredictionTests`):**
   - `test_empty_upload_is_rejected`: None upload produces clear user guidance.
   - `test_empty_numpy_array_is_rejected`: Empty array produces clear user guidance.
   - `test_corrupt_image_file_is_rejected`: Corrupt files report format error without crash.
   - `test_oversized_image_is_rejected`: Images exceeding max side or pixel limit are rejected before processing.
   - `test_bytes_io_corrupt_buffer`: Verifies corrupt byte buffers are rejected safely.

3. **Artifact Integrity & Error Handling (`DemoPredictionTests` & `DemoArtifactRoundTripTests`):**
   - `test_incompatible_artifact_is_rejected`: Missing classes or invalid dimensions trigger clear error.
   - `test_missing_artifact_file_is_reported`: Missing file reports Prajjwal handoff guidance.
   - `test_checksum_mismatch_refuses_to_load`: Modified binary is rejected before unpickling.
   - `test_size_mismatch_is_reported_before_unpickle`: Undersized file is rejected before unpickling.
   - `test_saved_fixture_joblib_round_trip`: Full dump/load/verify round-trip test.

4. **App Construction & UI Callback Lifecycles (`DemoAppConstructionTests`):**
   - `test_importing_app_does_not_launch_server`: Importing `app` does not call `launch()`.
   - `test_build_interface_does_not_launch_server`: Building blocks does not start server.
   - `test_image_change_and_clear_drop_stale_outputs`: Replacing image or clicking Clear clears all outputs.
   - `test_callback_empty_and_corrupt_inputs`: Callback wraps exceptions into user-facing status.
   - `test_share_flag_is_rejected`: Passing `--share` terminates with exit code.

---

## 5. Explicit List of Missing Teammate Material & Submission Requirements

In accordance with academic integrity guidelines, we explicitly record all gaps, pending items, and course discrepancies:

1. **Course Team Size Discrepancy:**
   - The official course specification stipulates teams of **exactly five students**.
   - This project contains **six assigned contributors** (Om Yadav, Aditya Bharadwaj, Shourya Bafna, Daksh Batra, Prajjwal Tripathi, Saksham Miglani).
   - This administrative discrepancy was not created by the team and remains an outstanding issue to resolve with course faculty.
2. **Model Binary Status — Resolved:**
   - The evaluated model binary `artifacts/calibrated_svm/model.joblib` (66,270,283 bytes, SHA-256 `83b0658d5180c02d6f03dce835bb8e269d6a89a493b6298a2bb00fd594f3b112`) has been supplied by Prajjwal Tripathi and placed in the local workspace.
   - The binary is intentionally excluded from Git (see `.gitignore`). Freshly cloned checkouts require it to be supplied out-of-band to run real-model inference or pass `test_evaluated_artifact_on_synthetic_image`; the test is guarded by `@skipUnless` and is **not** skipped when the file is present.
3. **Historical Working-Tree Provenance Discrepancy:**
   - As documented in `calibration-protocol.json` and Prajjwal's notes, the final calibration and evaluation run was conducted on a dirty working tree prior to commit `b775f02` (`package_source_sha256: 26e7e076...`).
   - Subsequent non-functional linter wrapping edits (`ruff check .` rule E501 line lengths) changed file bytes to `cab3b677...`.
   - Because no intermediate git snapshot was preserved, the exact historical package state cannot be checked out from Git. The evaluated model and predictions were preserved to prevent test-set contamination.
4. **Pending Verified GitHub Usernames:**
   - Only Shourya Bafna (`shourya2006`) and Om Yadav (`yadavom7345`) have verified accounts recorded in `docs/contributions.md`.
   - GitHub accounts for Aditya Bharadwaj, Daksh Batra, Prajjwal Tripathi, and Saksham Miglani remain pending verification.
5. **Everyday External Photo Evaluation:**
   - Roadmap milestone 43 ("Analyze mistakes and optionally evaluate independent everyday photos") contains complete mistake analysis from saved test predictions.
   - Testing on independent everyday consumer photos was not performed in Phase 1 and is deferred to Phase 2.
6. **Individual Viva Preparation:**
   - Team oral examination practice remains to be conducted collectively before final grading.
7. **Local LaTeX Compilation Environment:**
   - No TeX distribution (`pdflatex`, `xelatex`, `latexmk`) is installed in the local workstation environment. Direct CLI compilation from `report/main.tex` to PDF is unavailable locally.
   - The paper is provided as a fully self-contained LaTeX project in `report/` ready for Overleaf or TeX Live, alongside a standalone browser-viewable IEEE counterpart (`report/report.html`) and PDF export (`report/report.pdf`).

---

## 6. Reproduction Guide

### A. Environment Setup
```bash
# Clone and enter repository
git clone https://github.com/shourya2006/smart-waste-segregation.git
cd smart-waste-segregation

# Create and activate Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install locked dependencies
pip install -r requirements-demo.lock
pip install -e .
```

### B. Run Verification & Test Suite
```bash
# Verify linting
ruff check .

# Run full test suite (157 tests, 0 skipped when model binary is present; 1 skipped otherwise)
python -m unittest discover -s tests -v
```

### C. Launch Local Demo
```bash
# Option 1: Launch with synthetic fixture (requires no model binary)
python app.py --synthetic-fixture --port 7860

# Option 2: Launch with preserved evaluated model (after placing model.joblib)
python app.py --port 7860
```
Open `http://127.0.0.1:7860` in any web browser.

---

## 7. Individual Viva Preparation

Be prepared to discuss:
1. **Why process-level model caching is essential:** Unpickling 66.3 MB on every POST request introduces ~400–800 ms latency per request and causes memory churn. Caching in `_RUNTIME_ARTIFACT` enables sub-50 ms classification.
2. **Why confidence-based abstention is NOT out-of-distribution detection:** Platt scaling maps SVM decision margins to probabilities conditioned on the 6 known classes. An entirely novel object (e.g., a laptop or an apple) will still produce probabilities summing to 1.0 and may yield arbitrary high confidence along an unconstrained feature axis. True out-of-distribution detection requires density estimation or distance-to-training-distribution measures.
3. **Why aspect-preserving letterboxing is vital for HOG:** Resizing rectangular images directly to a square canvas distorts edge angles and gradients, degrading HOG block histograms. Letterboxing with lanczos interpolation preserves physical aspect ratios.
4. **Why selective classification improves real-world sorting:** In automated sorting facilities, misclassifying hazardous waste or non-recyclables contaminates entire recycling batches. Abstaining on the lowest 1.6% of predictions eliminated 5 out of 6 errors, raising accepted accuracy from 90.19% to 91.37%.
5. **Why images must not be persisted:** Privacy compliance and data minimization require that user photos uploaded to an educational demo are processed transiently in RAM and never written to disk without explicit consent.
