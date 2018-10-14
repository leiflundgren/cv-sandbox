import numpy
import cv2
from common import trace

casc_file_frontal = "cascades/haarcascade_frontalface_default.xml"
casc_file_glasses = 'cascades/haarcascade_eye_tree_eyeglasses.xml'

class EyeChecker:
    def __init__(self):
        self.load_cascades()
        pass

    def load_cascades(self):
        # Create the haar cascade
        self.face_frontal_casc = cv2.CascadeClassifier(casc_file_frontal)
        self.glasses_casc = cv2.CascadeClassifier(casc_file_glasses)
        trace(6, 'Cascades loaded')

    class Check:
        def __init__(self, im):
            self.im = im
        pass

    def check_image(self, image_thing) -> Check:
        
        if isinstance(image_thing, str):
            trace(6, 'loading img ' + image_thing )
            img = cv2.imread(image_thing)
            trace(6, 'loaded img ' + image_thing + ' --> ' + str(type(img)))
            return Check(img)
        else:
            trace(2, 'cannot load image from ', image_thing)

    pass

