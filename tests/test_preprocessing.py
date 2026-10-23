"""Behavior checks for the shared image path used by features and inference."""

import io
import tomllib
import unittest
from pathlib import Path

from PIL import Image

from waste_sorting.preprocessing import preprocess_image


def config(width=128, height=128, padding=(255, 255, 255)):
    return {
        "preprocessing": {
            "width": width,
            "height": height,
            "preserve_aspect_ratio": True,
            "padding_rgb": list(padding),
        }
    }


class PreprocessingTests(unittest.TestCase):
    def test_project_configuration_produces_expected_rgb_size(self):
        settings = tomllib.loads(Path("configs/phase1.toml").read_text())
        result = preprocess_image(Image.new("RGB", (512, 384), "red"), settings)
        self.assertEqual(result.size, (128, 128))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((64, 15)), (255, 255, 255))
        self.assertEqual(result.getpixel((64, 16)), (255, 0, 0))
        self.assertEqual(result.getpixel((64, 111)), (255, 0, 0))
        self.assertEqual(result.getpixel((64, 112)), (255, 255, 255))

    def test_portrait_padding_preserves_aspect_ratio(self):
        result = preprocess_image(Image.new("RGB", (20, 40), "red"), config(10, 10))
        self.assertEqual(result.getpixel((1, 5)), (255, 255, 255))
        self.assertEqual(result.getpixel((2, 5)), (255, 0, 0))
        self.assertEqual(result.getpixel((6, 5)), (255, 0, 0))
        self.assertEqual(result.getpixel((7, 5)), (255, 255, 255))

    def test_odd_padding_pixel_is_on_bottom(self):
        result = preprocess_image(Image.new("RGB", (5, 2), "red"), config(5, 5))
        self.assertEqual(result.getpixel((2, 0)), (255, 255, 255))
        self.assertEqual(result.getpixel((2, 1)), (255, 0, 0))
        self.assertEqual(result.getpixel((2, 2)), (255, 0, 0))
        self.assertEqual(result.getpixel((2, 3)), (255, 255, 255))
        self.assertEqual(result.getpixel((2, 4)), (255, 255, 255))

    def test_square_grayscale_input(self):
        result = preprocess_image(Image.new("L", (8, 8), 73), config(8, 8))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (73, 73, 73))

    def test_alpha_is_composited_on_configured_color(self):
        result = preprocess_image(
            Image.new("RGBA", (4, 4), (255, 0, 0, 128)), config(4, 4, (0, 0, 255))
        )
        self.assertEqual(result.getpixel((0, 0)), (128, 0, 127))

    def test_fully_transparent_input_uses_background(self):
        result = preprocess_image(
            Image.new("RGBA", (2, 2), (0, 0, 0, 0)), config(4, 4, (10, 20, 30))
        )
        self.assertEqual(result.getextrema(), ((10, 10), (20, 20), (30, 30)))

    def test_palette_transparency_is_supported(self):
        image = Image.new("P", (2, 2), 0)
        image.putpalette([0, 0, 0] * 256)
        image.info["transparency"] = 0
        result = preprocess_image(image, config(2, 2, (11, 22, 33)))
        self.assertEqual(result.getpixel((0, 0)), (11, 22, 33))

    def test_exif_orientation_applied_without_changing_input(self):
        image = Image.new("RGB", (4, 2), "red")
        image.getexif()[274] = 6
        before = image.tobytes()
        result = preprocess_image(image, config(4, 4))
        self.assertEqual(result.getpixel((0, 2)), (255, 255, 255))
        self.assertEqual(result.getpixel((1, 2)), (255, 0, 0))
        self.assertEqual(image.size, (4, 2))
        self.assertEqual(image.getexif()[274], 6)
        self.assertEqual(image.tobytes(), before)
        self.assertIsNone(result.getexif().get(274))

    def test_output_is_deterministic_and_independent(self):
        image = Image.new("RGB", (13, 7), "green")
        original = image.tobytes()
        first = preprocess_image(image, config())
        second = preprocess_image(image, config())
        self.assertIsNot(first, second)
        self.assertEqual(first.tobytes(), second.tobytes())
        first.putpixel((0, 0), (0, 0, 0))
        self.assertEqual(image.tobytes(), original)
        self.assertNotEqual(first.tobytes(), second.tobytes())

    def test_very_thin_image_retains_one_pixel(self):
        result = preprocess_image(Image.new("RGB", (1000, 1), "red"), config(8, 8))
        self.assertEqual(result.getpixel((4, 3)), (255, 0, 0))
        self.assertEqual(result.getpixel((4, 2)), (255, 255, 255))

    def test_invalid_input_and_configuration(self):
        for image in (None, b"bad", "image.jpg"):
            with self.subTest(image=image), self.assertRaises(TypeError):
                preprocess_image(image, config())
        image = Image.new("RGB", (2, 2))
        with self.assertRaises(TypeError):
            preprocess_image(image, None)
        with self.assertRaises(ValueError):
            preprocess_image(image, {})
        for value in (0, -1, 2.5, True, "128"):
            for key in ("width", "height"):
                settings = config()
                settings["preprocessing"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    preprocess_image(image, settings)
        for padding in ([1, 2], [1, 2, 256], [1, -1, 3], [1, 2, True], "red"):
            settings = config()
            settings["preprocessing"]["padding_rgb"] = padding
            with self.subTest(padding=padding), self.assertRaises(ValueError):
                preprocess_image(image, settings)

    def test_stretching_is_rejected(self):
        settings = config()
        settings["preprocessing"]["preserve_aspect_ratio"] = False
        with self.assertRaisesRegex(ValueError, "preserve_aspect_ratio"):
            preprocess_image(Image.new("RGB", (4, 2)), settings)

    def test_closed_lazy_image_has_clear_error(self):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 2), "red").save(buffer, format="PNG")
        buffer.seek(0)
        image = Image.open(buffer)
        buffer.close()
        with self.assertRaisesRegex(ValueError, "decode"):
            preprocess_image(image, config())


if __name__ == "__main__":
    unittest.main()
