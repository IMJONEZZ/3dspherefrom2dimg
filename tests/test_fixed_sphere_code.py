import unittest
import numpy as np
import torch
import cv2
from fixed_sphere_code import (
    SphereModel, DifferentiableRenderer,
    extract_silhouette, initialize_sphere_params
)

class TestFixedSphereCode(unittest.TestCase):
    def setUp(self):
        # Synthetic test image: white circle on black background
        self.image_size = (256, 256)
        self.image = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(self.image, (128, 128), 50, (255, 255, 255), -1)
        self.silhouette = extract_silhouette(self.image)
        self.params = initialize_sphere_params(self.silhouette)

    def test_silhouette_shape_and_range(self):
        self.assertEqual(self.silhouette.shape, self.image_size)
        self.assertTrue(np.all((self.silhouette >= 0) & (self.silhouette <= 1)))

    def test_initialize_params_structure(self):
        keys = {'radius', 'stretch', 'rotation', 'translation'}
        self.assertTrue(keys.issubset(set(self.params)))
        self.assertEqual(len(self.params['stretch']), 3)
        self.assertEqual(len(self.params['rotation']), 3)
        self.assertEqual(len(self.params['translation']), 3)

    def test_sphere_model_forward(self):
        model = SphereModel([
            self.params['radius'],
            self.params['stretch'],
            self.params['rotation'],
            self.params['translation']
        ])
        points = torch.rand((100, 3), dtype=torch.float32)
        output = model(points)
        self.assertEqual(output.shape, (100, 3))
        self.assertTrue(torch.is_tensor(output))

    def test_renderer_output_shape_and_range(self):
        model = SphereModel([
            self.params['radius'],
            self.params['stretch'],
            self.params['rotation'],
            self.params['translation']
        ])
        renderer = DifferentiableRenderer(image_size=self.image_size)
        output = renderer.render_silhouette(model, resolution=60)
        self.assertEqual(output.shape, self.silhouette.shape)
        self.assertTrue(torch.is_tensor(output))
        self.assertTrue((output >= 0).all())
        self.assertTrue((output <= 1).all())

if __name__ == '__main__':
    unittest.main()
