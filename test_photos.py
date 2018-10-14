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
        trace(2, 'setUp: phototests running')
        self.eye_checker = EyeChecker()
        trace(7, 'created eye-check')

    def tearDown(self):
        trace(2,'tearDown: phototests done')

    #def test_load_spice_girls(self):
    #    trace(5, 'testing spice girls image')
    #    m = self.eye_checker.check_image('sample_photos/spice-girls.jpg')
    #    self.assertIsNotNone(m, 'check_image should return possitive')
    #    #m.show_faces()
    #    self.assertEqual(4, len(m.faces)) # Sporty spice leans head to side?

    def test_load_frontal_glasses(self):
        file = 'sample_photos/frontal.jpg'
        trace(5, 'testing ' + file)
        m = self.eye_checker.check_image(file)
        m.show_faces()

if __name__ == '__main__':
    unittest.main()

