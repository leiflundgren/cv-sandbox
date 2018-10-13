import unittest
from eye_check import EyeChecker
import os
import common 

class phototests(unittest.TestCase):
    """description of class"""

    def setUp(self):
        common.tracelevel = 9  
        common.trace(3, 'phototests running')
        self.eye_checker = EyeChecker()
        common.trace(7, 'created eye-check')

    def test_load_image(self):
        common.trace(5, 'testing frontal image')
        m = self.eye_checker.check_image('sample_photos/frontal.jpg')

if __name__ == '__main__':
    unittest.main()

