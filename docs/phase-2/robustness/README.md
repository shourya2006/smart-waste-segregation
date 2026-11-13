# Phase 2 robustness evaluation: everyday photos

**Status: not run yet.** The photos are being collected and audited
([docs/robustness](../../robustness/README.md)). No results exist yet, and none are reported.

When the audit is clean, the evaluation runs once:

```sh
waste-phase2-eval robustness --output docs/phase-2/robustness
```

That command replaces this page with the results: macro F1, accuracy, per-class recall,
coverage and accepted accuracy at the frozen demo thresholds, bootstrap 95% CIs, McNemar tests,
accuracy by background, lighting and device, and the comparison with the TrashNet test split.
It also writes `robustness-predictions.csv`, `robustness-metrics.json`, the confusion matrices
and `report/robustness-section.tex`. It refuses to run a second time.

## Limitations

- **Small set.** A few hundred photos, about 50 per class, so confidence intervals are wide
  and per-condition cells smaller still; cells under 10 photos are marked as small.
- **Collected by students.** Photos were taken by the team at home and on campus, not a
  random sample of real waste streams; backgrounds, items and phones reflect what we had.
- **Visual material labels.** Labels are the visible main material (TrashNet's six classes),
  assigned by the photographer; they are not local recycling rules, and mixed or dirty items
  were avoided rather than tested.
- **Unpaired comparison.** TrashNet test and robustness images are different sets, so the
  change in accuracy is a difference of two estimates, not a paired test.
- **Correlated conditions.** Background, lighting and phone were not fully crossed.

## Viva notes

1. **Q: Why does an external set matter when we already have a held-out test split?**
   **A:** The TrashNet test split comes from the same distribution as training: one
   photographer, a white posterboard, studio-like lighting. A held-out split measures how
   well the model generalises *within* that distribution. Everyday photos on tables, floors,
   cloth and grass under daylight, indoor and dim light, taken with different phones, are a
   distribution shift (covariate shift in background, illumination and sensor). Only an
   external set shows how much of the TrashNet accuracy survives that shift.
2. **Q: Why must the robustness set never be used for tuning?**
   **A:** As soon as a choice (a threshold, a temperature, a model, an augmentation) is made
   because it scores better on these photos, the photos stop being an unbiased estimate of
   performance on new everyday photos and the reported number becomes optimistic. It would
   also contradict the frozen protocol: models, temperatures and thresholds were committed in
   `protocol.json` before any test or robustness data was scored. That is why the command
   scores the set once and refuses to run again.
3. **Q: If accuracy drops, why not retrain with these photos?**
   **A:** That would be a new Phase 3 experiment needing its own fresh external set. Using
   these photos for training would leave no untouched everyday-photo estimate at all.
4. **Q: Are the condition breakdowns conclusive?**
   **A:** No. Groups are small and conditions are correlated (for example dim light may
   mostly be indoors on one phone), so they show where errors concentrate, not causes.
5. **Q: Does abstention detect out-of-distribution photos?**
   **A:** Not reliably. Coverage at the frozen demo threshold shows how often the models
   became unsure; a confident wrong answer on a new background is still possible.
