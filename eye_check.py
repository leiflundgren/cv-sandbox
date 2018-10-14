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

            self.eyes = []
            for i in range(0, len(self.faces)):                
                face = self.faces[i]
                (x,y,w,h) = face
                trace(8, 'detecting {0} face {1} shape {2}'.format(self.name, i, face))

                roi_gray = self.gray[y:y+h, x:x+w]
                roi_color = self.im[y:y+h, x:x+w]

                eyes = self.parent.glasses_casc.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.12,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                trace(6, 'eye-detecting in {0}, face {1} found {2} eyes: {3} \n{4} '.format(self.name, i, len(eyes), type(eyes), eyes))
                if len(eyes) > 0 :
                    try:
                        for (ex,ey,ew,eh) in eyes.tolist():
                            e2 = (ex+x, ey+y, ew, eh)
                            self.eyes.append(e2)
                    except:
                        trace(1, 'Failed to lambda on \n' + str(eyes))
                        raise
            trace(7, 'eye detect done\n' + str(self.eyes))

        def show_faces(self):
            trace(6, 'show faces on ' + self.name)
            img = self.im.copy()
            # Draw a rectangle around the faces
            for (x, y, w, h) in self.faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (40, 255, 0), 3)
                
            # and eyes
            for (x, y, w, h) in self.eyes:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 40, 0), 3)
                


            height, width, depth = img.shape
            max_h = 800.0
            max_w = 1000.0
            if height > max_h or width > max_w:
                scale = max_h / height if height > max_h else max_w/width
                #nw,nh= int(img.shape[1]*scale), int(img.shape[0]*scale)
                nw,nh= int(width*scale), int(height*scale)
                trace(7, 'Scaling image from {0} to {1}'.format((width,height), (nw,nh)))
                img = cv2.resize(img, (nw,nh))
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

