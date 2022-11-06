import cv2
import mediapipe as mp
import numpy as np


class face_summetry():
    def __init__(self):
        # Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode = True)

    def draw_face_landmarks(self,img):
        # Facial landmarks
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


        if not results.multi_face_landmarks == None:
            landmarks = results.multi_face_landmarks[0]

            for landmark in landmarks.landmark:
                x = landmark.x
                y = landmark.y
                
                relative_x = int(img.shape[1] * x)
                relative_y = int(img.shape[0] * y)
                
                cv2.circle(img, (relative_x, relative_y),1, (0, 0, 255), 1)

        return img

    def head_tilt_detection(self,image):
        head_moved = False

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False
        
        # Get the result
        results = self.face_mesh.process(image)
        
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, _ = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
            

                # See where the user's head tilting
                if x < -2 or x >2 or y>2 or y<-2:
                    print('Head tilted')
                    head_moved = True
                


        return head_moved

    def main_loop(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, img = cap.read()

            if not ret:
                print('=============Camera not found================')
                break

            if not self.head_tilt_detection(img):
                self.draw_face_landmarks(img)

            else:
                print('=========Keep the head straight========')

            cv2.imshow('Live feed',img)
            cv2.waitKey(10)



objectt = face_summetry()
objectt.main_loop()
