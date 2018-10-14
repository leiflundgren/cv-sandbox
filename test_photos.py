""" Test open-cv detect things in a photo """

import unittest
from eye_check import EyeChecker
import os
import common
from common import trace
#from src.common import tracelevel

class phototests(unittest.TestCase):
    """description of class"""

    def setUp(self):
        common.tracelevel = 9  
        trace(2, 'phototests running')
        self.eye_checker = EyeChecker()
        trace(7, 'created eye-check')

    def tearDown(self):
        trace(2,'phototests done')

    def test_load_image(self):
        trace(5, 'testing frontal image')
        m = self.eye_checker.check_image('sample_photos/frontal.jpg')
        self.assertIsNotNone(m, 'check_image should return possitive')

if __name__ == '__main__':
    unittest.main()

