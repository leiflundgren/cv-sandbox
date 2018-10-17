import unittest

# Install cv2 though pip
# pip install opencv-python 

import cv2

class OpenCVTests(unittest.TestCase):
    def test_version2(self):

        self.assertTrue(cv2.__version__.startswith('3'))

if __name__ == '__main__':
    unittest.main()
