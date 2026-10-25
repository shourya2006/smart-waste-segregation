"""Local Gradio photo-upload demo for Phase 1 waste classification.

Loads the preserved evaluated artifact once per process and calls
predict_image for every classification. Importing this module does not
start a server.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from waste_sorting.demo import (
    DEFAULT_ARTIFACT_PATH,
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_ARTIFACT_SIZE,
    FROZEN_ABSTENTION_THRESHOLD,
    LIMITATIONS_NOTE,
    DemoError,
    DemoView,
    classify_upload,
    empty_view,
    evaluated_artifact_status,
    inject_runtime_artifact,
    load_evaluated_artifact,
)
from waste_sorting.extended_features import DIMENSION as EXTENDED_DIMENSION
from waste_sorting.prediction import CANONICAL_CLASSES

PROBABILITY_HEADERS = ["Category", "Calibrated probability"]
SYNTHETIC_BANNER = (
    "Synthetic fixture mode is active. Displayed scores are not predictions "
    "from the preserved evaluated model."
)

CUSTOM_CSS = """
.uncertain-banner:has(textarea:placeholder-shown) {
    display: none;
}
.uncertain-banner textarea, .uncertain-banner input, .uncertain-banner {
    background: #fff4d6 !important;
    border: 2px solid #8a6d1d !important;
    color: #5c4808 !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
}
.primary-result textarea {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}
.limitations-box {
    font-size: 0.95rem;
}
.synthetic-warning-box {
    background: #fff3cd !important;
    border: 2px solid #e0a800 !important;
    color: #856404 !important;
    padding: 10px 14px !important;
    border-radius: 6px !important;
    font-size: 1rem !important;
    margin-bottom: 10px !important;
}
"""


def view_to_outputs(view: DemoView) -> tuple:
    return (
        view.error_message,
        view.primary_result,
        view.uncertainty_message,
        view.confidence_text,
        view.probability_rows,
        view.tentative_class,
    )


def clear_outputs() -> tuple:
    return view_to_outputs(empty_view())


def classify_callback(image) -> tuple:
    try:
        return view_to_outputs(classify_upload(image))
    except DemoError as exc:
        return view_to_outputs(empty_view(error_message=str(exc)))


def on_image_change(image) -> tuple:
    """Drop previous predictions whenever the upload is cleared or replaced."""
    del image
    return clear_outputs()


def clear_all() -> tuple:
    return (None, *clear_outputs())


def build_interface(synthetic_fixture: bool = False) -> gr.Blocks:
    """Construct the Gradio demo interface.
    Returns a Gradio Blocks object without launching.
    """
    with gr.Blocks(title="Smart Waste Segregation", analytics_enabled=False) as demo:
        if synthetic_fixture:
            gr.Markdown(SYNTHETIC_BANNER, elem_classes=["synthetic-warning-box"])
        gr.Markdown("# Smart Waste Segregation Demo")
        gr.Markdown(LIMITATIONS_NOTE, elem_classes=["limitations-box"])
        image_input = gr.Image(label="Upload Photo", type="numpy")
        with gr.Row():
            classify_btn = gr.Button("Classify")
            clear_btn = gr.Button("Clear")
        error_msg = gr.Textbox(label="Status", interactive=False, elem_id="demo-status")
        primary_res = gr.Textbox(
            label="Result", interactive=False, elem_classes=["primary-result"]
        )
        uncertainty_msg = gr.Textbox(
            label="Uncertainty", interactive=False, elem_classes=["uncertain-banner"]
        )
        confidence_txt = gr.Textbox(label="Confidence", interactive=False, lines=3)
        prob_table = gr.Dataframe(
            headers=PROBABILITY_HEADERS, datatype=["str", "str"], interactive=False
        )
        tentative_cls = gr.Textbox(label="Tentative Class", interactive=False)
        outputs = [
            error_msg, primary_res, uncertainty_msg,
            confidence_txt, prob_table, tentative_cls,
        ]
        prediction_event = classify_btn.click(
            classify_callback, inputs=image_input, outputs=outputs, api_name="classify"
        )
        image_input.change(
            on_image_change, inputs=image_input, outputs=outputs,
            cancels=[prediction_event], queue=False,
        )
        clear_btn.click(
            clear_all, outputs=[image_input, *outputs],
            cancels=[prediction_event], queue=False, api_name="clear",
        )
    return demo

def _synthetic_fixture_artifact(abstain: bool = False) -> dict:
    import numpy as np

    probabilities = (
        np.array([0.18, 0.16, 0.17, 0.19, 0.15, 0.15], dtype=np.float64)
        if abstain
        else np.array([0.04, 0.03, 0.78, 0.05, 0.06, 0.04], dtype=np.float64)
    )

    class FixtureModel:
        def __init__(self, probs):
            self.classes_ = np.arange(len(CANONICAL_CLASSES))
            self.n_features_in_ = EXTENDED_DIMENSION
            self.probabilities = probs

        def predict_proba(self, features):
            return np.tile(self.probabilities, (len(features), 1))

    return {
        "calibrated_model": FixtureModel(probabilities),
        "classes": list(CANONICAL_CLASSES),
        "threshold": FROZEN_ABSTENTION_THRESHOLD,
        "input_dimension": EXTENDED_DIMENSION,
        "model_type": "synthetic_fixture",
        "base_model": "not_the_evaluated_model",
    }






def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the local Smart Waste Segregation photo demo."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Path to the preserved evaluated model.joblib file.",
    )
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help=(
            "Use an in-memory fixture for interface checks. "
            "Do not treat fixture scores as evaluated-model predictions."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Disabled by default. Do not enable for public deployment.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.share:
        raise SystemExit("Public sharing is disabled for this local educational demo.")
    if args.synthetic_fixture:
        inject_runtime_artifact(_synthetic_fixture_artifact())
        print(SYNTHETIC_BANNER)
    else:
        status = evaluated_artifact_status(args.artifact)
        if status != "verified":
            print(
                "Evaluated model artifact is not verified "
                f"({status} at {args.artifact}; expected SHA-256 "
                f"{EXPECTED_ARTIFACT_SHA256}, {EXPECTED_ARTIFACT_SIZE} bytes). "
                "The interface will still start; classification will report the error. "
                "Obtain the original file from Prajjwal Tripathi. "
                "Do not train a replacement."
            )
        else:
            load_evaluated_artifact(args.artifact)
            print(f"Loaded evaluated artifact from {args.artifact}")
    demo = build_interface(synthetic_fixture=args.synthetic_fixture)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=False,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
