import unittest

import numpy as np

from inverse_kinematics import PlatformConfig, StewartPlatformIK


class StewartPlatformIKTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ik = StewartPlatformIK()

    def test_home_pose_has_six_finite_valid_extensions(self) -> None:
        extensions = self.ik.solve(np.zeros(3), np.zeros(3))

        self.assertEqual(extensions.shape, (6,))
        self.assertTrue(np.isfinite(extensions).all())
        self.assertTrue(self.ik.extensions_are_valid(extensions))

    def test_home_pose_regression_values(self) -> None:
        extensions = self.ik.solve(np.zeros(3), np.zeros(3))

        np.testing.assert_allclose(
            extensions,
            np.array(
                [
                    0.0003205151491571,
                    0.00032833525961162,
                    0.00032833525961162,
                    0.0003205151491571,
                    0.00032546848269022,
                    0.00032546848269022,
                ]
            ),
            atol=1e-9,
        )

    def test_rotation_matrix_is_orthonormal(self) -> None:
        rotation = self.ik.rotation_matrix([0.1, -0.2, 0.3])

        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_rotation_matrix_uses_rx_ry_rz_order(self) -> None:
        roll, pitch, yaw = 0.1, -0.2, 0.3

        rotation_x = self.ik._rotation_x(roll)
        rotation_y = self.ik._rotation_y(pitch)
        rotation_z = self.ik._rotation_z(yaw)

        np.testing.assert_allclose(
            self.ik.rotation_matrix([roll, pitch, yaw]),
            rotation_x @ rotation_y @ rotation_z,
        )

    def test_invalid_translation_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "translation_m must have shape"):
            self.ik.solve([0.0, 0.0], [0.0, 0.0, 0.0])

    def test_invalid_rotation_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only finite values"):
            self.ik.solve([0.0, 0.0, 0.0], [0.0, np.nan, 0.0])

    def test_config_rejects_invalid_geometry_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "base_joints must have shape"):
            PlatformConfig(base_joints=np.zeros((3, 5)))


if __name__ == "__main__":
    unittest.main()
