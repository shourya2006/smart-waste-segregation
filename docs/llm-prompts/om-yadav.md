# Om Yadav — data verification and preprocessing

Follow shared-rules.md in this directory. Branch: `feat/preprocessing`.
Review partner: Aditya Bharadwaj.

## Implement

1. Independently reproduce the existing image-integrity checks and verify the
   committed manifest/split summaries. Inspect the six uncertain duplicate pairs
   with locally generated contact sheets and record your own observations.
   Preserve the conservative grouping and current manifest; propose any correction
   explicitly rather than silently changing the dataset.
2. Add src/waste_sorting/preprocessing.py with
   `preprocess_image(image, config)`. Read the preprocessing section of the full
   TOML mapping. Correct EXIF orientation, support grayscale and RGBA inputs,
   convert to RGB, preserve aspect ratio, resize with documented interpolation,
   and center on the configured padding color. Composite transparency onto the
   configured background. Defaults are a white 128×128 output. Do not mutate
   the caller's image or introduce random augmentation.
3. Fail clearly on invalid image objects, invalid dimensions, or invalid padding.
   Document the rounding and centering policy for uneven padding.
4. Add meaningful tests in tests/test_preprocessing.py for EXIF orientation,
   landscape/portrait aspect ratio, exact output size and mode, alpha compositing,
   padding, deterministic output, and preservation of the input image.

## Acceptance and handoff

Verify both synthetic edge cases and representative training images. Existing
19 data tests must continue to pass, along with new checks. Keep source photos
and generated preview sheets out of Git. Provide Aditya with the import path,
input/output contract, and examples. Write docs/member-notes/om-yadav.md with
verification evidence, preprocessing rationale, and viva points. Commit your
implementation and tests, push your branch, and open a PR. Do not retrain models.
