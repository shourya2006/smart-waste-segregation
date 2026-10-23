# Saksham Mighlani — demo and submission integration

Follow shared-rules.md in this directory. Branch: `feat/demo-integration`.
Review partner: Prajjwal Tripathi. Real predictions depend on his artifact and
`predict_image(image, artifact)` interface.

## Implement

1. Build app.py with a local Gradio photo upload interface. Load the trusted
   project artifact once and call the shared prediction function. Display an
   image preview, material prediction, calibrated confidence, optional top-three
   probabilities, and the uncertain response when abstained. Preserve the
   artifact's class/probability ordering; do not independently compute confidence.
2. Handle missing model files, unreadable inputs, no input, and unsupported images
   with useful messages. Without a real artifact, show a clear model-unavailable
   state. Test fixtures may support development but must not appear as real output.
3. Show a visible note: this educational model predicts visual material classes;
   local recycling rules, contamination, and mixed materials affect disposal.
   Do not assign universal bin colors or disposal instructions from the label.
4. Add integration tests for valid predictions, abstention, probability/label
   alignment, missing artifacts, and failed inputs. Verify a real saved-model
   round trip with Prajjwal when available. Document setup and launch commands
   and coordinate optional Gradio dependency locking without unrelated upgrades.
5. Assemble a LaTeX/IEEE report structure and presentation outline from each
   member's actual technical handoff. Collect their own methodology, theory,
   literature, results, and figures. Preserve author attribution. Clearly mark
   unavailable sections as TODO; never invent experiments, citations, or scores.
   Do not claim report/presentation completion before the sections exist.
6. Write a clean-checkout reproduction guide covering acquisition, fixed manifests,
   features, model selection, calibration, final evaluation, and demo launch using
   commands actually supported by the integrated code. Record missing steps.

## Acceptance and handoff

Start the shell, missing-model handling, independent tests, and report structure
before the artifact is ready. Finish real integration once dependencies merge,
and verify successful, uncertain, and failed-input flows. Write your own technical
notes at docs/member-notes/saksham-mighlani.md. Commit your implementation/tests
and document assembly, push your branch, and open a PR. Do not deploy publicly,
train a substitute classifier, or rewrite other members' model pipelines.
