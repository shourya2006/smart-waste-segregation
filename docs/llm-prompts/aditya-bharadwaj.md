# Aditya Bharadwaj — handcrafted features

Follow shared-rules.md in this directory. Branch: `feat/image-features`.
Review partner: Om Yadav. Depends on Om's preprocessing interface.

## Implement

1. Add src/waste_sorting/features.py. Use Om's `preprocess_image` rather than
   writing another image-resizing path. Provide `extract_features(image, config)`
   and a clearly documented way to select descriptor families.
2. Implement separate normalized HSV histograms using 16/8/8 bins by default
   (32 values total). Define the channel ranges explicitly for the conversion
   library, and normalize each channel histogram consistently.
3. Implement grayscale HOG with the configured 9 orientations, 16×16 cells,
   2×2 cell blocks, and L2-Hys normalization. Default output is 1,764 values.
4. Implement uniform LBP on integer grayscale, P=8 and radius=1, with a normalized
   10-bin histogram by default. Concatenate HSV, HOG, LBP in that order: 1,806
   finite float32 values for the current full configuration. Derive dimensions
   from configuration rather than hardcoding one size throughout the code.
5. Support color-only, HOG-only, LBP-only, color+HOG, all-three, and removal of
   each family from the combined configuration for later comparisons.
6. Implement a reproducible feature-cache builder carrying image IDs, labels,
   duplicate groups, and split membership without losing alignment. Exclude
   quarantined rows. Cache keys/metadata must include source/manifest hashes,
   preprocessing settings, feature settings, and feature ordering. Refuse stale
   or misaligned caches. Caches remain local under data/processed/.

## Acceptance and handoff

Test dimensions, histogram normalization, finiteness, determinism, feature ordering,
configurable subsets, and cache invalidation. Extract the real included dataset
once preprocessing is available; report actual dimensions and runtime, not model
accuracy. Do not fit scaling/PCA or train models. Write a reproducible handoff at
docs/member-notes/aditya-bharadwaj.md for Shourya and Daksh, with descriptor theory
and examples. Commit implementation and tests, push your branch, and open a PR.
