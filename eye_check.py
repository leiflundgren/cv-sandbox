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
        def __init__(self, parent, imageName, im):
            trace(8, 'creating Check from ' + imageName)
            self.parent = parent
            self.name = imageName
            self.im = im
            self.gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
            # trace(8, im)

            # Detect faces in the image
            self.faces = self.parent.face_frontal_casc.detectMultiScale(
                self.gray,
                scaleFactor=1.12,
                minNeighbors=8,
                minSize=(50, 100)
            )
            trace(6, 'detecting {0} found {1} faces'.format(self.name, len(self.faces)))

        def show_faces(self):
            img = self.im.copy()
            # Draw a rectangle around the faces
            for (x, y, w, h) in self.faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (40, 255, 0), 2)
                

            cv2.imshow("Faces found", img)
            cv2.waitKey(0)

    def check_image(self, image_thing) : #-> self.Check:
        
        if isinstance(image_thing, str):
            trace(6, 'loading img ' + image_thing )
            img = cv2.imread(image_thing)
            trace(6, 'loaded img ' + image_thing + ' --> ' + str(type(img)))
            return self.Check(self, image_thing, img)
        else:
            trace(2, 'cannot load image from ', image_thing)

    pass

