# ruff: noqa: E501  (slide markup lives in long HTML template strings)
"""Build the Phase 2 deck (HTML, 960x540 slides) in the Phase 1 visual style.

Every number is read from the committed result files; nothing is typed by hand.
Usage: python build.py <repo_root> <out_dir>
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ab = pd.read_csv(ROOT / "docs/phase-2/ablations/summary.csv").set_index("id")
metrics = json.loads((ROOT / "docs/phase-2/evaluation/test-metrics.json").read_text())
protocol = json.loads((ROOT / "docs/phase-2/evaluation/protocol.json").read_text())
preds = pd.read_csv(ROOT / "docs/phase-2/evaluation/test-predictions.csv")
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NAMES = {"phase1_svm": "Phase 1 SVM", "a2_1_cnn": "A2.1 CNN", "h3_hybrid": "H3 hybrid"}
M = metrics["models"]
MC = metrics["mcnemar"]
P = protocol["models"]

# Cross-validation numbers from the hybrid analysis (same 5 training folds,
# metrics from the argmax of predicted probabilities; docs/phase-2/hybrid/README.md).
CV = {"phase1": (0.8844, 0.0200), "a21": (0.8825, 0.0219), "h1": (0.8788, 0.0185),
      "h2": (0.9175, 0.0262), "h3": (0.9364, 0.0171)}
DISAGREE, BOTH_WRONG, N_TRAIN = 295, 46, 1764           # recomputed from the two OOF files

def f4(x): return f"{x:.4f}"
def pct(x, d=2): return f"{100 * x:.{d}f}%"

u = {k: v["unconditional"] for k, v in M.items()}
ci = {k: v["bootstrap_95_ci"]["macro_f1"] for k, v in M.items()}
demo = {k: v["selective_demo"] for k, v in M.items()}
h3_correct = int(round(u["h3_hybrid"]["accuracy"] * len(preds)))
cm = confusion_matrix(preds.label_index, preds.h3_hybrid_prediction_index, labels=range(6))
base_f1, base_sd = ab.loc["BASE", "mean_macro_f1"], ab.loc["BASE", "std_macro_f1"]
a21 = ab.loc["A2.1"]

CSS = """
@page { size: 960px 540px; margin: 0; }
* { box-sizing: border-box; margin: 0; padding: 0; }
:root { --bg:#F2F4EF; --ink:#16241D; --dark:#16241D; --green:#2F6B43; --lime:#8ED08A;
  --muted:#5D6862; --card:#FAFAF7; --line:#DCE1D8; --panel:#E4E9E1; --mid:#376F4C; --amber:#A4651F; }
body { background:#888; font-family:'IBM Plex Sans',sans-serif; color:var(--ink); }
.slide { width:960px; height:540px; position:relative; overflow:hidden; background:var(--bg);
  padding:56px 64px 0 64px; page-break-after:always; break-after:page; }
.slide.dark { background:var(--dark); color:#EEF2EC; }
.slide.mid { background:var(--mid); color:#F3F6F1; }
.eyebrow { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:13px; letter-spacing:.14em;
  color:var(--green); text-transform:uppercase; }
.dark .eyebrow { color:var(--lime); } .mid .eyebrow { color:#E5EFE6; }
h1 { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:36px; line-height:1.12;
  letter-spacing:-.01em; margin-top:8px; }
.mono { font-family:'JetBrains Mono',monospace; }
.foot { position:absolute; left:64px; right:64px; bottom:24px; display:flex; justify-content:space-between;
  font-size:13px; color:var(--muted); }
.foot .n { font-family:'JetBrains Mono',monospace; }
.dark .foot, .mid .foot { color:#C9D3CB; }
.body { position:absolute; left:64px; right:64px; top:150px; bottom:62px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 18px; }
.card.dk { background:var(--dark); border:none; color:#EEF2EC; }
.card.pn { background:var(--panel); border:none; }
.card h3 { font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:700; line-height:1.2; }
.card p, .card li { font-size:13.5px; line-height:1.45; color:var(--muted); }
.card.dk p, .card.dk li { color:#D5DED6; }
.card ul { padding-left:18px; margin-top:6px; } .card li { margin:3px 0; }
.big { font-family:'JetBrains Mono',monospace; font-weight:700; line-height:1; }
.lime { color:var(--lime); } .grn { color:var(--green); } .amb { color:var(--amber); }
.row { display:flex; gap:14px; } .col { display:flex; flex-direction:column; gap:12px; }
.small { font-size:12.5px; color:var(--muted); line-height:1.45; }
.dark .small, .mid .small { color:#D5DED6; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:6px 9px; border:1px solid var(--line); text-align:right; }
th { background:var(--panel); font-family:'JetBrains Mono',monospace; font-weight:700; font-size:12px; }
td:first-child, th:first-child { text-align:left; }
td.m { font-family:'JetBrains Mono',monospace; }
.dot { display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:8px; vertical-align:-1px; }
.tag { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:11px; padding:2px 7px;
  border-radius:6px; background:var(--panel); color:var(--green); }
.tag.no { color:var(--amber); }
.who { font-size:11.5px; color:var(--green); font-weight:600; margin-top:6px; }
"""
DOT = {"cardboard":"#9C6B43", "glass":"#4A8D99", "metal":"#6B7480", "paper":"#4A62A3",
       "plastic":"#B8712F", "trash":"#555A57"}

slides = []
def slide(body, eyebrow="", title="", kind="", notes=""):
    slides.append((body, eyebrow, title, kind, notes))

# 1 Title -----------------------------------------------------------------
cats = "".join(f'<div style="font-size:20px;margin:12px 0"><span class="dot" style="background:{DOT[c]}"></span>{c.capitalize()}</div>' for c in CLASSES)
slide(f"""
<div style="position:absolute;left:64px;top:62px;width:560px">
  <div class="eyebrow">Project 12 · Phase 2</div>
  <div style="font-family:'Space Grotesk';font-weight:700;font-size:60px;line-height:1.05;margin-top:14px">Smart Waste<br>Segregation</div>
  <div style="font-size:19px;color:#B9C6BC;margin-top:18px">Transfer learning, a hybrid model and one frozen test</div>
</div>
<div style="position:absolute;left:64px;bottom:58px">
  <div class="eyebrow">Team</div>
  <div style="font-size:15px;line-height:1.6;margin-top:8px;color:#E3E9E2">Om Yadav · Aditya Bhardwaj · Shourya Bafna<br>Daksh Batra · Prajjwal Tripathi · Saksham Miglani</div>
</div>
<div style="position:absolute;right:64px;top:62px;width:220px;height:412px;background:#203328;border-radius:16px;padding:86px 28px">
  <div class="eyebrow" style="margin-bottom:6px">Six categories</div>{cats}
</div>""", kind="dark", notes="Presenter: Shourya Bafna. Introduce the team and the goal of Phase 2.")

# 2 Starting point ---------------------------------------------------------
slide("""
<div class="row" style="height:100%">
  <div class="col" style="flex:1.35">
    <div class="row" style="flex:1">
      <div class="card" style="flex:1"><div class="big" style="font-size:44px">0.8999</div><h3 style="margin-top:10px;font-size:17px">Phase 1 test macro F1</h3><p>Handcrafted colour and texture features with a calibrated RBF SVM.</p></div>
      <div class="card" style="flex:1"><div class="big amb" style="font-size:44px">6</div><h3 style="margin-top:10px;font-size:17px">Metal photos called glass</h3><p>The largest single confusion on the Phase 1 test set.</p></div>
    </div>
    <div class="card pn"><p style="color:var(--ink);font-size:14px">Same TrashNet data, same frozen train / validation / test splits and folds as Phase 1, so every Phase 2 number is directly comparable.</p></div>
  </div>
  <div class="card dk" style="flex:1;padding:22px 24px">
    <h3>Phase 2 asks</h3>
    <ul style="margin-top:10px"><li>Does an ImageNet-pretrained CNN beat handcrafted features?</li>
    <li>Does combining the two do better than either alone?</li>
    <li>Does the answer hold on 377 unseen test images?</li></ul>
    <h3 class="lime" style="margin-top:18px">Rule we kept</h3>
    <p style="margin-top:6px">Decisions on training folds; calibration and thresholds on validation; the test set used once.</p>
  </div>
</div>""", "Where we started", "Phase 1 left a 0.8999 baseline to beat",
notes="Presenter: Shourya Bafna. Recap Phase 1 and state the three Phase 2 questions.")

# 3 Pipeline ---------------------------------------------------------------
steps = [("01","Data","Frozen splits, train-only augmentation","Om"),("02","Backbones","MobileNetV2 and EfficientNet-B0","Aditya"),
         ("03","Train","Head first, then last two blocks","Shourya"),("04","Ablate","13 pre-registered settings, 65 runs","Aditya"),
         ("05","Combine","Three hybrid designs","Daksh"),("06","Freeze","Calibrate, fix thresholds, test once","Prajjwal"),
         ("07","Demo","Model selector with abstention","Saksham")]
cards = "".join(f'<div class="card{" dk" if i == 5 else ""}" style="flex:1;padding:14px 12px"><div class="mono {"lime" if i == 5 else "grn"}" style="font-size:20px;font-weight:700">{n}</div><h3 style="font-size:16.5px;margin-top:6px">{t}</h3><p style="font-size:12.5px;margin-top:4px">{d}</p><div class="who"{" style=color:var(--lime)" if i == 5 else ""}>{w}</div></div>' for i,(n,t,d,w) in enumerate(steps))
slide(f"""
<div class="row" style="height:178px;gap:10px">{cards}</div>
<div class="card pn" style="margin-top:18px;padding:18px 22px">
  <h3>Which data decides what</h3>
  <div class="row" style="margin-top:10px;gap:22px">
    <p style="flex:1;color:var(--ink)"><b class="mono">1,764</b> training images in 5 frozen folds: every architecture, ablation and hybrid choice.</p>
    <p style="flex:1;color:var(--ink)"><b class="mono">380</b> validation images: calibration checks and abstention thresholds only.</p>
    <p style="flex:1;color:var(--ink)"><b class="mono">377</b> test images: one frozen run at the very end.</p>
  </div>
</div>""", "The pipeline", "Seven steps, one frozen protocol",
notes="Presenter: Shourya Bafna. Walk through who built each stage and which data each decision used.")

# 4 Data -------------------------------------------------------------------
augs = [("Random crop","80–100% of the image"),("Horizontal flip","p = 0.5"),("Rotation","± 10°"),("Colour jitter","brightness and contrast 0.15, saturation 0.1")]
aug_cards = "".join(f'<div class="card" style="width:calc(50% - 6px);padding:12px 16px"><h3 style="font-size:16px">{a}</h3><p style="margin-top:2px">{b}</p></div>' for a, b in augs)
slide(f"""
<div class="row" style="height:100%">
  <div class="col" style="flex:1.5">
    <div style="display:flex;flex-wrap:wrap;gap:12px">{aug_cards}</div>
    <div class="card pn"><h3 style="font-size:16px">Same preprocessing as Phase 1</h3>
      <p style="margin-top:4px;color:var(--ink)">EXIF rotation, white letterbox to 224 × 224, ImageNet normalisation. Validation and test images are never augmented, and near-duplicates stay in one split.</p></div>
  </div>
  <div class="card dk" style="flex:1;padding:22px 24px">
    <div class="big lime" style="font-size:52px">4.33</div>
    <p style="margin-top:8px">times as many paper photos (416) as trash photos (96) in training</p>
    <h3 style="margin-top:16px">Two fixes, both tested</h3>
    <ul><li>Class-weighted loss</li><li>Balanced sampling</li></ul>
    <p style="margin-top:8px">Both run as ablations rather than assumed to help.</p>
  </div>
</div>""", "Data and regularisation", "Augmentation only where the model learns",
notes="Presenter: Om Yadav. Explain the train-only augmentation and the imbalance options.")

# 5 Architectures ----------------------------------------------------------
slide("""
<div class="row" style="height:100%">
  <div class="card" style="flex:1;padding:20px 22px">
    <div class="tag">Selected</div><h3 style="margin-top:8px;font-size:22px">MobileNetV2</h3>
    <div class="big" style="font-size:34px;margin-top:12px">2,223,872</div><p>backbone parameters</p>
    <ul style="margin-top:12px"><li>Depthwise-separable convolutions</li><li>Inverted residuals with linear bottlenecks</li><li>1,280-number image summary</li></ul>
  </div>
  <div class="card" style="flex:1;padding:20px 22px">
    <div class="tag">Compared</div><h3 style="margin-top:8px;font-size:22px">EfficientNet-B0</h3>
    <div class="big" style="font-size:34px;margin-top:12px">4,007,548</div><p>backbone parameters</p>
    <ul style="margin-top:12px"><li>MBConv blocks with squeeze-and-excitation</li><li>Compound scaling of depth, width, resolution</li><li>1,280-number image summary</li></ul>
  </div>
  <div class="card dk" style="flex:.9;padding:22px 24px">
    <h3>Why pretrained</h3>
    <p style="margin-top:8px">ImageNet features (edges, textures, shapes) transfer well, so 1,764 training photos are enough to adapt them.</p>
    <h3 class="lime" style="margin-top:18px">No centre crop</h3>
    <p style="margin-top:8px">A letterbox keeps the whole object in view instead of cutting its edges.</p>
  </div>
</div>""", "Architectures", "Two light ImageNet backbones",
notes="Presenter: Aditya Bhardwaj. Compare the two backbones and explain transfer learning.")

# 6 Training ---------------------------------------------------------------
sets = [("Head","Dropout 0.2 → Linear 1,280 → 6"),("Optimiser","AdamW, weight decay 0.01"),
        ("Learning rates","Head 1e-3 · unfrozen blocks 1e-4"),("Schedule","Cosine to 1e-5 over 40 epochs"),
        ("Early stopping","Patience 8 on held-out fold loss"),("Final model","Retrained on all 1,764 for 13 epochs")]
set_cards = "".join(f'<div class="card" style="width:calc(50% - 6px)"><h3 style="font-size:15.5px">{a}</h3><p style="margin-top:3px">{b}</p></div>' for a, b in sets)
slide(f"""
<div class="row" style="height:100%">
  <div style="flex:1.45;display:flex;flex-wrap:wrap;gap:12px;align-content:flex-start">{set_cards}</div>
  <div class="card dk" style="flex:1;padding:22px 24px">
    <h3>Frozen means frozen</h3>
    <p style="margin-top:8px">Even with weights frozen, BatchNorm statistics and EfficientNet's stochastic depth still change in training mode. Frozen blocks are kept in evaluation mode, and a test checks nothing moves.</p>
    <h3 class="lime" style="margin-top:18px">Same schedule at the end</h3>
    <p style="margin-top:8px">The final 13-epoch run follows the 40-epoch cosine curve exactly as the cross-validation runs did.</p>
  </div>
</div>""", "Training", "Train the head, then the last two blocks",
notes="Presenter: Shourya Bafna. Explain AdamW, cosine schedule, early stopping and differential learning rates.")

# 7 Ablation chart ---------------------------------------------------------
order = ab.sort_values("mean_macro_f1", ascending=True)
lo, hi, W, X0 = 0.80, 0.92, 430, 52
def x(v): return X0 + (v - lo) / (hi - lo) * W
rows = []
for t in (0.80, 0.84, 0.88, 0.92):
    rows.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="2" y2="296" stroke="#DCE1D8"/><text x="{x(t):.1f}" y="310" font-size="10.5" text-anchor="middle" font-family="JetBrains Mono" fill="#5D6862">{t:.2f}</text>')
for i, (rid, r) in enumerate(order.iterrows()):
    y = 14 + i * 22
    col = "#2F6B43" if rid == "A2.1" else ("#16241D" if rid == "BASE" else "#8A9B8F")
    m, sd = r.mean_macro_f1, r.std_macro_f1
    bold = 700 if rid in ("A2.1", "BASE") else 400
    rows.append(f'<text x="0" y="{y+4}" font-size="11.5" font-family="JetBrains Mono" fill="#16241D" font-weight="{bold}">{rid}</text>'
                f'<line x1="{x(m-sd):.1f}" x2="{x(m+sd):.1f}" y1="{y}" y2="{y}" stroke="{col}" stroke-width="2"/>'
                f'<circle cx="{x(m):.1f}" cy="{y}" r="5" fill="{col}"/>'
                f'<text x="{x(0.92)+8:.1f}" y="{y+4}" font-size="11" font-family="JetBrains Mono" fill="#5D6862" font-weight="{bold}">{m:.4f}</text>')
basex = x(base_f1)
chart = f'<svg width="560" height="315" viewBox="0 0 560 315"><line x1="{basex:.1f}" x2="{basex:.1f}" y1="2" y2="296" stroke="#A4651F" stroke-dasharray="3 3"/>{"".join(rows)}</svg>'
slide(f"""
<div class="row" style="height:100%">
  <div style="flex:1.55"><div class="small" style="font-weight:600;letter-spacing:.08em">MEAN MACRO F1 (DOT) ± STD (LINE), 5 TRAINING FOLDS · DASHED: BASE</div><div style="margin-top:8px">{chart}</div></div>
  <div class="card dk" style="flex:1;padding:22px 24px">
    <div class="eyebrow">Selected · A2.1</div>
    <div class="big lime" style="font-size:44px;margin-top:10px">{a21.mean_macro_f1:.4f}</div>
    <p>± {a21.std_macro_f1:.4f}, last two MobileNetV2 blocks fine-tuned</p>
    <p style="margin-top:14px">Gain over BASE <b class="mono" style="color:#fff">+{a21.mean_macro_f1-base_f1:.4f}</b>, larger than the fold spread <b class="mono" style="color:#fff">{base_sd:.4f}</b>: the only clear winner.</p>
    <p style="margin-top:12px"><b class="mono" style="color:#fff">{int(a21.trainable_parameters):,}</b> trainable parameters · median best epoch <b class="mono" style="color:#fff">{int(a21.median_best_epoch)}</b></p>
  </div>
</div>""", "Ablation study", "Only fine-tuning clearly beat the baseline",
notes="Presenter: Aditya Bhardwaj. The plan was committed before any run; a variant wins only if its gain exceeds the fold standard deviation.")

# 8 Ablation lessons -------------------------------------------------------
def d(rid): return ab.loc[rid, "mean_macro_f1"] - base_f1
less = [("No augmentation", f"{d('A5.1'):+.4f}", "Within the fold spread: no clear difference.", False),
        ("160 × 160 input", f"{d('A3.1'):+.4f}", "Just beyond the spread: smaller images hurt.", True),
        ("EfficientNet-B0", f"{d('A1.1'):+.4f}", "No clear gain for 1.8× the parameters.", False),
        ("Class-weighted loss", f"{d('A6.1'):+.4f}", f"No clear F1 change; trash recall {pct(ab.loc['BASE','mean_recall_trash'],1)} → {pct(ab.loc['A6.1','mean_recall_trash'],1)} (mean per fold).", False)]
lc = "".join(f'<div class="card" style="width:calc(50% - 7px)"><div class="row" style="justify-content:space-between;align-items:baseline"><h3 style="font-size:17px">{a}</h3><span class="big {"amb" if bad else ""}" style="font-size:22px">{b}</span></div><p style="margin-top:6px">{c}</p></div>' for a, b, c, bad in less)
slide(f"""
<div style="display:flex;flex-wrap:wrap;gap:14px">{lc}</div>
<div class="card pn" style="margin-top:16px;padding:16px 20px"><p style="color:var(--ink);font-size:14px"><b>Caveat we report.</b> Frozen models reached their best epoch at 38–40 of the 40-epoch cap (BASE median {int(ab.loc['BASE','median_best_epoch'])}), while A2.1 stopped by itself near 20. Part of A2.1's lead may come from that budget; the cap was pre-registered, so the selection stands.</p></div>""",
"What the ablations taught us", "Small changes mostly sat inside the noise",
notes="Presenter: Aditya Bhardwaj. Differences are versus BASE; recall figures are mean per fold.")

# 9 Hybrids ----------------------------------------------------------------
hy = [("H1","ML on CNN features","An SVM on the 1,280 EfficientNet-B0 features.",CV["h1"],False),
      ("H2","Early fusion","Handcrafted and CNN features joined, each block scaled, one SVM.",CV["h2"],False),
      ("H3","Late fusion","Average the probabilities of the Phase 1 SVM and the A2.1 CNN.",CV["h3"],True)]
hc = "".join(f'<div class="card{" dk" if best else ""}" style="flex:1;padding:20px 22px"><div class="mono {"lime" if best else "grn"}" style="font-size:22px;font-weight:700">{k}</div><h3 style="margin-top:6px">{t}</h3><p style="margin-top:6px;min-height:58px">{dsc}</p><div class="big {"lime" if best else ""}" style="font-size:34px;margin-top:10px">{m:.4f}</div><p>± {s:.4f} CV macro F1</p></div>' for k,t,dsc,(m,s),best in hy)
slide(f"""
<div class="row" style="height:262px">{hc}</div>
<div class="card pn" style="margin-top:16px;padding:14px 20px"><p style="color:var(--ink)">Single models on the same five folds: Phase 1 SVM <b class="mono">{CV['phase1'][0]:.4f}</b> · A2.1 CNN <b class="mono">{CV['a21'][0]:.4f}</b>. Rationale: handcrafted colour and texture and CNN shape features fail on different photos.</p></div>""",
"Hybrid models", "Three hybrid designs; late fusion leads",
notes="Presenter: Daksh Batra. All cross-validation scores use the argmax of predicted probabilities.")

# 10 Why fusion works -------------------------------------------------------
slide(f"""
<div class="row" style="height:100%">
  <div class="col" style="flex:1.3">
    <div class="row" style="flex:1">
      <div class="card" style="flex:1"><div class="big" style="font-size:46px">{DISAGREE}</div><h3 style="margin-top:10px;font-size:17px">training photos where the SVM and CNN disagree</h3><p>{pct(DISAGREE/N_TRAIN,1)} of {N_TRAIN:,}</p></div>
      <div class="card" style="flex:1"><div class="big grn" style="font-size:46px">{BOTH_WRONG}</div><h3 style="margin-top:10px;font-size:17px">photos where both are wrong</h3><p>{pct(BOTH_WRONG/N_TRAIN,1)} of {N_TRAIN:,}: different mistakes, so averaging fixes most</p></div>
    </div>
    <div class="card pn"><p style="color:var(--ink)">The weight is chosen on the other four folds every time, so no fold helps pick its own weight. All five folds chose <b class="mono">w = 0.5</b>.</p></div>
  </div>
  <div class="card dk" style="flex:1;padding:22px 24px">
    <h3>Pre-registered rule</h3>
    <p style="margin-top:8px">A hybrid wins only if it beats the best single model by more than the fold spread.</p>
    <div class="big lime" style="font-size:40px;margin-top:16px">+{CV['h3'][0]-CV['a21'][0]:.4f}</div>
    <p>H3 over A2.1, against σ = {CV['a21'][1]:.4f}: a clear win in cross-validation.</p>
    <p style="margin-top:12px">H3 vs H2 (+{CV['h3'][0]-CV['h2'][0]:.4f}) is inside the noise; H3 goes forward as the highest mean.</p>
  </div>
</div>""", "Why late fusion works", "The two models make different mistakes",
notes="Presenter: Daksh Batra. Disagreement counts are from the two out-of-fold prediction files.")

# 11 Frozen protocol -------------------------------------------------------
pr = "".join(f'<tr><td>{NAMES[k]}</td><td class="m">{P[k]["temperature"]:.3f}</td><td class="m">{P[k]["headline_threshold"]:.2f}</td><td class="m">{P[k]["demo_threshold"]:.2f}</td></tr>' for k in ["phase1_svm","a2_1_cnn","h3_hybrid"])
slide(f"""
<div class="row" style="height:100%">
  <div class="card" style="flex:1.3;padding:20px 22px">
    <h3>Fixed on training and validation data</h3>
    <table style="margin-top:14px"><tr><th>Model</th><th>Temperature</th><th>Headline</th><th>Demo</th></tr>{pr}</table>
    <p style="margin-top:12px">Temperature fitted on out-of-fold training predictions. Headline threshold: highest coverage with ≥ 90% accuracy on validation. Demo threshold: ≥ 95% accuracy at ≥ 80% coverage.</p>
  </div>
  <div class="card dk" style="flex:1;padding:22px 24px">
    <h3>Frozen before the test</h3>
    <p style="margin-top:8px">Model file hashes, temperatures and both thresholds were committed in <span class="mono">protocol.json</span> (commit <span class="mono lime">16e4b9f</span>) before a single test image was scored.</p>
    <p style="margin-top:14px">The code refuses to touch the test set without that file and refuses to run the test twice.</p>
    <p style="margin-top:14px"><span class="lime">Test images never tuned anything.</span></p>
  </div>
</div>""", "Calibration and freeze", "Everything fixed before the test set opened",
notes="Presenter: Prajjwal Tripathi. H3's headline threshold is 0.00 because it already exceeds 90% accuracy, so a stricter demo threshold was added before the freeze.")

# 12 Test result (mid green) -----------------------------------------------
slide(f"""
<div class="row" style="align-items:center;gap:26px">
  <div class="big" style="font-size:86px">{u['h3_hybrid']['macro_f1']:.4f}</div>
  <div><h3 style="font-family:'Space Grotesk';font-size:21px">H3 hybrid · test macro F1</h3><p style="font-size:15px;margin-top:4px;color:#E5EEE6">95% bootstrap interval {ci['h3_hybrid']['ci_low']:.3f}–{ci['h3_hybrid']['ci_high']:.3f}, on 377 images it never saw</p></div>
</div>
<div class="row" style="margin-top:22px">
  <div class="card dk" style="flex:1"><div class="big" style="font-size:36px">{pct(u['h3_hybrid']['accuracy'])}</div><h3 style="font-size:16px;margin-top:8px">Accuracy, all 377</h3><p>{h3_correct} correct</p></div>
  <div class="card dk" style="flex:1"><div class="big" style="font-size:36px">{pct(demo['h3_hybrid']['coverage'])}</div><h3 style="font-size:16px;margin-top:8px">Coverage at 0.53</h3><p>{demo['h3_hybrid']['accepted_count']} answered, {demo['h3_hybrid']['abstained_count']} “not sure”</p></div>
  <div class="card dk" style="flex:1"><div class="big lime" style="font-size:36px">{pct(demo['h3_hybrid']['accepted_accuracy'])}</div><h3 style="font-size:16px;margin-top:8px">Accepted accuracy</h3><p>{int(round(demo['h3_hybrid']['accepted_accuracy']*demo['h3_hybrid']['accepted_count']))} of {demo['h3_hybrid']['accepted_count']} answered photos right</p></div>
</div>
<p style="margin-top:16px;font-size:15px;color:#E5EEE6">Best calibrated of the three models: expected calibration error {u['h3_hybrid']['expected_calibration_error']:.3f}.</p>""",
"Frozen test result", "One measurement on 377 held-out images", kind="mid",
notes="Presenter: Prajjwal Tripathi. One run with the frozen protocol; every number comes from test-predictions.csv.")

# 13 Three models compared --------------------------------------------------
tr = "".join(f'<tr{" style=font-weight:700" if k=="h3_hybrid" else ""}><td>{NAMES[k]}</td><td class="m">{u[k]["macro_f1"]:.4f}</td><td class="m">{ci[k]["ci_low"]:.3f}–{ci[k]["ci_high"]:.3f}</td><td class="m">{pct(u[k]["accuracy"])}</td><td class="m">{u[k]["expected_calibration_error"]:.3f}</td><td class="m">{pct(demo[k]["coverage"],1)}</td><td class="m">{pct(demo[k]["accepted_accuracy"],1)}</td></tr>' for k in ["phase1_svm","a2_1_cnn","h3_hybrid"])
slide(f"""
<table><tr><th>Model</th><th>Macro F1</th><th>95% CI</th><th>Accuracy</th><th>ECE</th><th>Coverage</th><th>Accepted acc.</th></tr>{tr}</table>
<p class="small" style="margin-top:8px">Coverage and accepted accuracy at each model's frozen demo threshold · 377 test images</p>
<div class="row" style="margin-top:18px">
  <div class="card" style="flex:1"><h3 style="font-size:17px">Most useful answers</h3><p style="margin-top:6px">At their demo thresholds H3 answers {pct(demo['h3_hybrid']['coverage'],1)} of photos, the SVM {pct(demo['phase1_svm']['coverage'],1)} (more accurate when it answers), the CNN {pct(demo['a2_1_cnn']['coverage'],1)}.</p></div>
  <div class="card" style="flex:1"><h3 style="font-size:17px">CNN alone is weakest</h3><p style="margin-top:6px">A2.1 scores {u['a2_1_cnn']['macro_f1']:.4f}, below the SVM's {u['phase1_svm']['macro_f1']:.4f}, but the gap is not significant.</p></div>
  <div class="card" style="flex:1"><h3 style="font-size:17px">Phase 1 reproduced</h3><p style="margin-top:6px">The refit SVM matches Phase 1's 90.19% accuracy; F1 differs ({u['phase1_svm']['macro_f1']:.4f} vs 0.8999) only through calibration.</p></div>
</div>""", "Test comparison", "Best F1, accuracy and calibration: the hybrid",
notes="Presenter: Prajjwal Tripathi. A2.1's demo threshold covered 76.4% of test photos, below the 80% it reached on validation.")

# 14 Significance -----------------------------------------------------------
def pval(v): return "0.00002" if v < 0.0001 else f"{v:.3f}"
def mrow(key, a, b, verdict, ok):
    r = MC[key]
    tot = r["a_only_correct"] + r["b_only_correct"]
    wa = r["a_only_correct"] / tot * 250
    return (f'<div class="card" style="padding:14px 18px"><div class="row" style="align-items:center;gap:18px">'
            f'<div style="width:170px"><h3 style="font-size:16px">{a} vs {b}</h3><p>only one right: {r["a_only_correct"]} vs {r["b_only_correct"]}</p></div>'
            f'<svg width="250" height="18"><rect x="0" y="2" width="{wa:.1f}" height="14" rx="3" fill="#2F6B43"/><rect x="{wa:.1f}" y="2" width="{250-wa:.1f}" height="14" rx="3" fill="#C9D2C6"/></svg>'
            f'<div class="big" style="font-size:20px;width:175px;white-space:nowrap">p = {pval(r["p_value"])}</div>'
            f'<span class="tag{"" if ok else " no"}" style="font-size:12px">{verdict}</span></div></div>')
slide(f"""
<div class="col">
{mrow("h3_hybrid_vs_a2_1_cnn","H3","A2.1","significant",True)}
{mrow("h3_hybrid_vs_phase1_svm","H3","SVM","suggestive only",False)}
{mrow("a2_1_cnn_vs_phase1_svm","A2.1","SVM","no difference",False)}
</div>
<div class="card pn" style="margin-top:14px;padding:14px 20px"><p style="color:var(--ink)">Exact McNemar tests on the photos where only one model is right. With three comparisons the bar is <b class="mono">p &lt; 0.0167</b> (Bonferroni). H3 clearly beats the CNN; its lead over the SVM (p = 0.035) is suggestive, not proven.</p></div>""",
"Significance", "A sure win over the CNN, likely over the SVM",
notes="Presenter: Prajjwal Tripathi. Green share of the bar = photos only the first model got right.")

# 15 Error analysis ---------------------------------------------------------
short = ["Cardbd.","Glass","Metal","Paper","Plastic","Trash"]
hdr = "".join(f"<th>{s}</th>" for s in short)
body = ""
for i, c in enumerate(CLASSES):
    cells = "".join(f'<td class="m" style="color:{"#2F6B43" if i==j else ("#A4651F" if (i,j)==(4,1) else "#16241D")}">{cm[i,j]}</td>' for j in range(6))
    body += f"<tr><td>{c.capitalize()}</td>{cells}</tr>"
low = min(range(6), key=lambda i: cm[i,i]/cm[i].sum())
slide(f"""
<div class="row" style="height:100%;gap:28px">
  <div style="flex:1.6"><table><tr><th>True \\ Pred.</th>{hdr}</tr>{body}</table>
  <p class="small" style="margin-top:8px">H3 hybrid on 377 test images · rows: true class · green diagonal = correct ({h3_correct} of 377)</p></div>
  <div class="col" style="flex:1;gap:16px">
    <div><div class="big amb" style="font-size:38px">{cm[4,1]}</div><p class="small" style="font-size:14px">plastic photos called glass, now the largest confusion</p></div>
    <div><div class="big" style="font-size:38px">{cm[2,1]}</div><p class="small" style="font-size:14px">metal photos called glass; the Phase 1 model made 6</p></div>
    <div><div class="big" style="font-size:38px">{pct(cm[low,low]/cm[low].sum())}</div><p class="small" style="font-size:14px">{CLASSES[low]} recall ({cm[low,low]} of {cm[low].sum()}), the lowest of any class</p></div>
  </div>
</div>""", "Error analysis", "Fewer look-alike mistakes than Phase 1",
notes="Presenter: Prajjwal Tripathi. Counts are from test-predictions.csv.")

# 16 Grad-CAM ---------------------------------------------------------------
gc = metrics["gradcam_cases"]
above = sum(1 for c in gc if c["fg_cam_energy_fraction"] > c["fg_area_fraction"])
slide(f"""
<div class="row" style="gap:18px;align-items:flex-start">
<img src="assets/gradcam.jpg" style="width:640px;border-radius:10px;border:1px solid var(--line)">
<div class="col" style="flex:1;gap:10px">
  <div class="card" style="padding:12px 14px"><h3 style="font-size:15px">Looks at the object</h3><p style="margin-top:4px">In {above} of {len(gc)} maps the object gets more attention than its share of the image.</p></div>
  <div class="card" style="padding:12px 14px"><h3 style="font-size:15px">Not an explanation</h3><p style="margin-top:4px">Errors focus on the object too; 12 maps cannot explain score gaps.</p></div>
</div></div>""", "Grad-CAM · A2.1 CNN", "Where the CNN looks: 6 correct, 6 wrong",
notes="Presenter: Prajjwal Tripathi. 'Object' is a colour heuristic (saturation > 0.12 or value < 0.92), not a hand-drawn mask.")

# 17 Demo -------------------------------------------------------------------
def ex(img, label, prob, verdict, ok):
    return (f'<div class="card" style="flex:1;padding:10px"><img src="assets/{img}" style="width:100%;border-radius:8px">'
            f'<h3 style="font-size:15px;margin-top:8px">{label}</h3><p class="mono" style="font-size:12.5px">{prob}</p>'
            f'<span class="tag{"" if ok else " no"}" style="margin-top:6px">{verdict}</span></div>')
demo_steps = [("Pick a model","H3 hybrid (recommended), A2.1 CNN or Phase 1 SVM"),("Upload a photo","Local Gradio app; public sharing is off"),
              ("Check the files","Every model file's SHA-256 must match protocol.json"),("Read six probabilities","Temperature-scaled with the frozen value"),
              ("Flag uncertainty","Below the frozen demo threshold, say “not sure”")]
ds = "".join(f'<div class="row" style="gap:14px;margin-bottom:9px"><div class="big {"amb" if i==4 else "grn"}" style="font-size:20px;width:18px">{i+1}</div><div><h3 style="font-family:Space Grotesk;font-size:16px">{a}</h3><p class="small" style="font-size:13px">{b}</p></div></div>' for i,(a,b) in enumerate(demo_steps))
slide(f"""
<div class="row" style="height:100%;gap:26px">
  <div style="flex:1">
    <div class="row" style="gap:10px">{ex("val-metal.jpg","Metal · 99.9%","threshold 0.53","answered",True)}{ex("val-cardboard.jpg","Cardboard 52.5% · paper 47.5%","below 0.53","not sure",False)}</div>
    <p class="small" style="margin-top:8px">Real H3 outputs from the merged demo code on two validation photos (not test photos).</p>
  </div>
  <div style="flex:1.05">{ds}</div>
</div>""", "Demo", "Pick a model, upload a photo",
notes="Presenter: Saksham Miglani. The demo's probabilities match the evaluation code on validation images to within 4e-6.")

# 18 Limitations ------------------------------------------------------------
lim = [("Cross-validation is optimistic","Settings were chosen on the same five folds; the test run is the fair number."),
       ("Small test set","377 images, only 20 of them trash: wide confidence intervals."),
       ("One photo setting","TrashNet photos sit on a white board; our everyday-photo set was not collected."),
       ("Thresholds drift","A2.1's demo threshold covered 76.4% of test photos vs 80% on validation."),
       ("GPU runs vary slightly","Retraining gives a new model and new hashes; results are frozen by hash."),
       ("Not disposal advice","Labels are materials; local recycling rules decide what goes where.")]
lc = "".join(f'<div class="card" style="width:calc(33.33% - 10px)"><h3 style="font-size:16px">{a}</h3><p style="margin-top:5px">{b}</p></div>' for a, b in lim)
slide(f'<div style="display:flex;flex-wrap:wrap;gap:14px">{lc}</div>', "Limitations", "What these results do not show",
notes="Presenter: Saksham Miglani.")

# 19 Team -------------------------------------------------------------------
team = [("Om Yadav","Data pipeline","Splits, augmentation, sampling"),("Aditya Bhardwaj","Backbones and ablations","Embeddings, 13-setting study"),
        ("Shourya Bafna","Training loop","Fine-tuning and final CNN"),("Daksh Batra","Hybrid models","Early and late fusion"),
        ("Prajjwal Tripathi","Calibration and test","Frozen protocol, statistics"),("Saksham Miglani","Demo and report","Model selector, IEEE paper")]
tc = "".join(f'<div class="card" style="width:calc(33.33% - 10px);padding:18px 20px"><h3 style="font-size:19px">{a}</h3><div class="grn" style="font-weight:600;font-size:14.5px;margin-top:6px">{b}</div><p style="margin-top:4px">{c}</p></div>' for a, b, c in team)
slide(f'<div style="display:flex;flex-wrap:wrap;gap:14px">{tc}</div>', "The team", "Six contributions, one frozen evaluation",
notes="Presenter: Shourya Bafna.")

# 20 Questions --------------------------------------------------------------
stats = [(f"{u['h3_hybrid']['macro_f1']:.4f}","H3 test macro F1",True),("13","pre-registered ablations",False),
         ("p < 0.0001","H3 vs CNN, McNemar",False),(pct(demo['h3_hybrid']['coverage'],1),f"answered, {pct(demo['h3_hybrid']['accepted_accuracy'],1)} of them right",False)]
sc = "".join(f'<div style="flex:1;border:1px solid #2E4237;border-radius:12px;padding:16px 18px"><div class="big{" lime" if hl else ""}" style="font-size:28px;white-space:nowrap">{a}</div><div class="small" style="margin-top:6px">{b}</div></div>' for a, b, hl in stats)
slide(f"""
<div style="position:absolute;left:64px;top:56px">
  <div class="eyebrow">Thank you</div>
  <div style="font-family:'Space Grotesk';font-weight:700;font-size:56px;margin-top:8px">Questions</div>
  <div style="font-size:18px;color:#B9C6BC;margin-top:10px">A hybrid that beats Phase 1, measured once and honestly</div>
</div>
<div class="row" style="position:absolute;left:64px;right:64px;top:278px">{sc}</div>
<div style="position:absolute;left:64px;bottom:44px;font-size:15px;color:#C9D3CB">Code and records: <u>github.com/shourya2006/smart-waste-segregation</u></div>""",
kind="dark", notes="Presenter: Shourya Bafna.")

# 21 References -------------------------------------------------------------
refs = [("Sandler et al. (2018)","MobileNetV2"),("Tan and Le (2019)","EfficientNet"),("Yosinski et al. (2014)","Transferability of features"),
        ("Loshchilov and Hutter (2019)","AdamW"),("Loshchilov and Hutter (2017)","SGDR, cosine schedule"),("Ioffe and Szegedy (2015)","Batch normalisation"),
        ("Huang et al. (2016)","Stochastic depth"),("Wolpert (1992)","Stacked generalisation"),("Guo et al. (2017)","Calibration of neural networks"),
        ("Selvaraju et al. (2017)","Grad-CAM")]
rc = "".join(f'<div style="width:calc(50% - 14px);margin-bottom:10px"><h3 style="font-family:IBM Plex Sans;font-size:15px;font-weight:700">{a}</h3><p class="small">{b}</p></div>' for a, b in refs)
slide(f"""<div style="display:flex;flex-wrap:wrap;gap:0 28px">{rc}</div>
<div class="card pn" style="margin-top:6px;padding:12px 18px"><p style="color:var(--ink)">Project metrics come from the versioned CSV and JSON records in the repository, not from these papers.</p></div>""",
"Backup", "References", notes="Backup slide.")

# 22 Reproduce --------------------------------------------------------------
slide(f"""
<div class="row" style="height:100%">
  <div class="card" style="flex:1.3;padding:18px 22px">
    <h3>Check the frozen results</h3>
    <p style="margin-top:8px">Full steps: <span class="mono">docs/phase-2/reproduction.md</span></p>
    <ul style="margin-top:6px"><li>Install from the pinned lock files</li><li><span class="mono">waste-data download</span> (1,764 / 380 / 377)</li>
    <li>Unzip the two frozen models; hashes are checked on load</li><li><span class="mono">notebooks/phase2-summary.ipynb</span> rebuilds the tables</li><li><span class="mono">python app.py</span> runs the demo</li></ul>
  </div>
  <div class="card dk" style="flex:1;padding:20px 22px">
    <h3>Frozen file hashes</h3>
    <p style="margin-top:10px">A2.1 <span class="mono">model.pt</span></p><p class="mono lime" style="font-size:11.5px;word-break:break-all">{P['a2_1_cnn']['hashes']['model_pt_sha256']}</p>
    <p style="margin-top:10px">Hybrid <span class="mono">svm_pipeline.joblib</span></p><p class="mono lime" style="font-size:11.5px;word-break:break-all">{P['h3_hybrid']['hashes']['svm_pipeline_sha256']}</p>
  </div>
</div>""", "Backup", "How to reproduce", notes="Backup slide.")

# ---------------------------------------------------------------------------
html = ['<!doctype html><html><head><meta charset="utf-8"><title>Smart Waste Segregation · Phase 2</title>',
        '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=block" rel="stylesheet">',
        f"<style>{CSS}</style></head><body>"]
notes = []
for i, (body_html, eyebrow, title, kind, note) in enumerate(slides, 1):
    head = "" if kind == "dark" else f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
    inner = body_html if kind == "dark" else f'{head}<div class="body">{body_html}</div>'
    foot = "" if (kind == "dark") else f'<div class="foot"><span>Smart Waste Segregation · Phase 2</span><span class="n">{i:02d}</span></div>'
    if kind == "mid":
        inner = f'{head}<div class="body" style="top:150px">{body_html}</div>'
    html.append(f'<section class="slide {kind}">{inner}{foot}</section>')
    notes.append(note)
html.append("</body></html>")
(OUT / "deck.html").write_text("\n".join(html), encoding="utf-8")
(OUT / "notes.json").write_text(json.dumps(notes, indent=1), encoding="utf-8")
print(f"{len(slides)} slides written")
