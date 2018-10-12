import cv2

casc_file_frontal = "cascades/haarcascade_frontalface_default.xml"
casc_file_glasses = 'cascades/haarcascade_eye_tree_eyeglasses.xml'

class EyeCheck:
    def __init__(self):
        self.load_cascades()
        pass

    def load_cascades(self):
        # Create the haar cascade
        self.face_frontal_casc = cv2.CascadeClassifier(casc_file_frontal)
        self.glasses_casc = cv2.CascadeClassifier(casc_file_glasses)


    pass

