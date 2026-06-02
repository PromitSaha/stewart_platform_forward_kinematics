import unittest

from inverse_kinematics.generate_dataset import (
    EXTENSION_COLUMNS,
    NOISY_EXTENSION_COLUMNS,
    generate_rows,
)


class DatasetGenerationTests(unittest.TestCase):
    def test_generated_rows_are_reproducible_and_physically_bounded(self) -> None:
        kwargs = {
            "samples": 10,
            "noise_std_mm": 0.5,
            "seed": 7,
            "max_attempt_multiplier": 100,
        }

        rows, _ = generate_rows(**kwargs)
        repeated_rows, _ = generate_rows(**kwargs)

        self.assertEqual(rows, repeated_rows)
        self.assertEqual(len(rows), 10)
        for row in rows:
            for column in EXTENSION_COLUMNS + NOISY_EXTENSION_COLUMNS:
                self.assertGreaterEqual(row[column], 0.0)
                self.assertLessEqual(row[column], 0.202)


if __name__ == "__main__":
    unittest.main()
