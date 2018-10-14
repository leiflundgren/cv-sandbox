import numpy
import cv2
from common import trace

casc_file_frontal = 'cascades/haarcascade_frontalface_default.xml'
casc_file_frontal_impr = 'cascades/lbpcascade_frontalface_improved.xml'
casc_file_profile_lbp = 'cascades/lbpcascade_profileface.xml'

casc_face_files = [ 
    'cascades/haarcascade_frontalface_alt.xml',
    'cascades/haarcascade_frontalface_alt2.xml',
    'cascades/haarcascade_frontalface_alt_tree.xml',
    'cascades/haarcascade_frontalface_default.xml',
    'cascades/lbpcascade_frontalcatface.xml',
    'cascades/lbpcascade_frontalface.xml',
    'cascades/lbpcascade_frontalface_improved.xml',
    'cascades/lbpcascade_profileface.xml'
]

casc_file_glasses = 'cascades/haarcascade_eye_tree_eyeglasses.xml'
casc_file_righteye = 'cascades/haarcascade_righteye_2splits.xml'
casc_file_lefteye = 'cascades/haarcascade_lefteye_2splits.xml'

class EyeChecker:
    def __init__(self):
        self.load_cascades()
        pass

    @staticmethod
    def load_cascade(filename):
        try:
            casc = cv2.CascadeClassifier(filename)
            trace(7, 'successfully loaded cascade from ' + filename)
            return casc
        except:
            trace(1, 'Failed to load "' + filename + '"')
            raise

    def load_cascades(self):
        # Create the haar cascade
        #self.face_frontal_casc = EyeChecker.load_cascade(casc_file_frontal)
        #self.face_frontal_impr_casc = EyeChecker.load_cascade(casc_file_frontal_impr)
        #self.face_profile_lbp_casc = EyeChecker.load_cascade(casc_file_profile_lbp)

        self.glasses_casc = EyeChecker.load_cascade(casc_file_glasses)
        self.righteye = EyeChecker.load_cascade(casc_file_righteye)
        self.lefteye = EyeChecker.load_cascade(casc_file_lefteye)

        l = lambda filename : (filename, EyeChecker.load_cascade(filename))
        self.face_cascades = map( l, casc_face_files)
        #[('frontal', self.face_frontal_casc), ('frontal_impr', self.face_frontal_impr_casc), ('profile', self.face_profile_lbp_casc)]

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
            self.detect_faces()

            self.eyes = []
            if len(self.faces) > 0:
                self.glasses = []
                self.detect_eyes_in_faces()
            # search for glasses everywhere!
            self.detect_glasses()
            # TODO, for glasses, detect eyes!!

            self.detect_right_left_eyes()

            #self.right_eyes = self.parent.righteye.detectMultiScale(
            #    self.gray,
            #    scaleFactor=1.12,
            #    minNeighbors=5,
            #    minSize=(30, 30)
            #)
            #trace(4, e)

            trace(7, 'detect done for ' + self.name)

        def detect_faces(self):
            faces_per_casc = {}
            self.faces = []
            for name, casc in self.parent.face_cascades:
                faces = casc.detectMultiScale(self.gray,
                    scaleFactor=1.12,
                    minNeighbors=8,
                    minSize=(50, 100))
                trace(6, 'detecting {0}/{1} found {2} faces'.format(self.name, name, len(faces)))
                faces_per_casc[name] = faces
                self.faces.extend(faces)

        def detect_eyes_in_faces(self):
            for i in range(0, len(self.faces)):                
                face = self.faces[i]
                (x,y,w,h) = face
                trace(8, 'detecting {0} face {1} shape {2}'.format(self.name, i, face))

                roi_gray = self.gray[y:y + h, x:x + w]
                roi_color = self.im[y:y + h, x:x + w]

                eyes = self.parent.glasses_casc.detectMultiScale(roi_gray,
                    scaleFactor=1.12,
                    minNeighbors=5,
                    minSize=(30, 30))
                trace(6, 'glasses-detecting in {0}, face {1} found {2} eyes: {3} \n{4} '.format(self.name, i, len(eyes), type(eyes), eyes))
                if len(eyes) > 0 :
                    try:
                        for (ex,ey,ew,eh) in eyes.tolist():
                            e2 = (ex + x, ey + y, ew, eh)
                            self.eyes.append(e2)
                    except:
                        trace(1, 'Failed to lambda on \n' + str(eyes))
                        raise

        def detect_glasses(self):
            self.glasses = self.parent.glasses_casc.detectMultiScale(self.gray,
                    scaleFactor=1.12,
                    minNeighbors=5,
                    minSize=(30, 30))
            trace(6, 'glasses-detecting in {0}, whole-image found {2} glasses: {3} \n{4} '.format(self.name, 'dummy', len(self.glasses), type(self.glasses), self.glasses))

        def detect_right_left_eyes(self):
            self.right_eyes = self.parent.righteye.detectMultiScale(self.gray,
                scaleFactor=1.12,
                minNeighbors=5,
                minSize=(30, 30))
            trace(6, 'right-eye-detecting in {0}, whole-image found {2} eyes: {3} \n{4} '.format(self.name, 'dummy', len(self.right_eyes), type(self.right_eyes), self.right_eyes))
            
            self.left_eyes = self.parent.lefteye.detectMultiScale(self.gray,
                scaleFactor=1.12,
                minNeighbors=5,
                minSize=(30, 30))
            trace(6, 'left-eye-detecting in {0}, whole-image found {2} eyes: {3} \n{4} '.format(self.name, 'dummy', len(self.left_eyes), type(self.left_eyes), self.left_eyes))

        def show_faces(self):
            trace(6, 'show faces on ' + self.name)
            img = self.im.copy()
            # Draw a rectangle around the faces
            #for (x, y, w, h) in self.faces:
            #    cv2.rectangle(img, (x, y), (x+w, y+h), (40, 255, 0), 3)
                
            ## and eyes
            #for (x, y, w, h) in self.eyes:
            #    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 40, 0), 3)
            
            ## and glasses
            #for (x, y, w, h) in self.glasses:
            #    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 40, 246), 3)
                 
            for (x, y, w, h) in self.left_eyes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (245, 40, 90), 3)
            for (x, y, w, h) in self.right_eyes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 40, 246), 3)

            height, width, depth = img.shape
            max_h = 800.0
            max_w = 1000.0
            if height > max_h or width > max_w:
                scale = max_h / height if height > max_h else max_w / width
                #nw,nh= int(img.shape[1]*scale), int(img.shape[0]*scale)
                nw,nh = int(width * scale), int(height * scale)
                trace(7, 'Scaling image from {0} to {1}'.format((width,height), (nw,nh)))
                img = cv2.resize(img, (nw,nh))
            cv2.imshow("Faces found", img)
            cv2.waitKey(0)

    def check_image(self, image_thing) : #-> self.Check:
        
        if isinstance(image_thing, str):
            trace(6, 'loading img ' + image_thing)
            img = cv2.imread(image_thing)
            trace(6, 'loaded img ' + image_thing + ' --> ' + str(type(img)))
            return self.Check(self, image_thing, img)
        else:
            trace(2, 'cannot load image from ', image_thing)

    pass

