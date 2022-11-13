import cv2
import mediapipe as mp
import numpy as np
import math
from time import time

class face_summetry():
    def __init__(self):
        # Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode = True)
        self.image_captured = False
        self.captured_image = ''
        self.display_text = ''

        self.start_time = time()
        self.end_time = time()
        self.timer_set = False
  
    def draw_face_landmarks(self,img):
        # Facial landmarks
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks == None:
            landmarks = results.multi_face_landmarks[0]

            for idx, landmark in enumerate(landmarks.landmark):
                

                if idx>0:
                    x = landmark.x
                    y = landmark.y
                    
                    relative_x = int(img.shape[1] * x)
                    relative_y = int(img.shape[0] * y)
                        

                    img = cv2.circle(img, (relative_x,relative_y), 2, (0,255,0), -1)
        


        return img

    def get_final_result(self,img):
        # Facial landmarks
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        horizontal_pts = []
        horizontal_pts_eyes = []
        vertical_pts = []

        left_lip_sym = []
        right_lip_sym = []
        left_eye_sym = []
        right_eye_sym = []



        if not results.multi_face_landmarks == None:
            landmarks = results.multi_face_landmarks[0]

            
            h,w,_ = img.shape

            for idx, landmark in enumerate(landmarks.landmark):

                if idx == 13:
                    x = landmark.x
                    y = landmark.y
                    
                    relative_x = int(img.shape[1] * x)
                    relative_y = int(img.shape[0] * y)

                    img = cv2.line(img, (relative_x,0), (relative_x,h), (0,0,0), 1)

                    for pts in range(0,h,5):
                        vertical_pts.append((relative_x,pts))


                #Eyes
                if idx == 246:
                    x = landmark.x
                    y = landmark.y
                    
                    relative_x = int(img.shape[1] * x)
                    relative_y = int(img.shape[0] * y)

                    img = cv2.line(img, (0,relative_y), (w,relative_y), (0,0,0), 1)

                    for pts in range(0,w,5):
                        horizontal_pts_eyes.append((pts,relative_y))


                #Lips
                if idx == 308:
                    x = landmark.x
                    y = landmark.y
                    
                    relative_x = int(img.shape[1] * x)
                    relative_y = int(img.shape[0] * y)

                    img = cv2.line(img, (0,relative_y), (w,relative_y), (0,0,0), 1)

                    for pts in range(0,w,5):
                        horizontal_pts.append((pts,relative_y))



            for idx, landmark in enumerate(landmarks.landmark):
                
                #Left lip
                if idx == 308:
                    x = landmark.x
                    y = landmark.y
                    
                    relative_x = int(img.shape[1] * x)
                    relative_y = int(img.shape[0] * y)


                    for h_points in horizontal_pts:
                        left_lip_sym.append(math.hypot(h_points[0] - relative_x, h_points[1] - relative_y))
                        

                    img = cv2.circle(img, (relative_x,relative_y), 2, (0,0,255), -1)

                #Right lip
                if idx == 61:
                    x = landmark.x
                    y = landmark.y
                    
                    relative_x = int(img.shape[1] * x)
                    relative_y = int(img.shape[0] * y)


                    for h_points in horizontal_pts:
                        right_lip_sym.append(math.hypot(h_points[0] - relative_x, h_points[1] - relative_y))

                    img = cv2.circle(img, (relative_x,relative_y), 2, (0,0,255), -1)
                
                #Right eye
                if idx == 246:
                    x = landmark.x
                    y = landmark.y
                    
                    relative_x = int(img.shape[1] * x)
                    relative_y = int(img.shape[0] * y)


                    for h_points in horizontal_pts_eyes:
                        right_eye_sym.append(math.hypot(h_points[0] - relative_x, h_points[1] - relative_y))

                    img = cv2.circle(img, (relative_x,relative_y), 2, (0,0,255), -1)

                #Left eye
                if idx == 466:
                    x = landmark.x
                    y = landmark.y
                    
                    relative_x = int(img.shape[1] * x)
                    relative_y = int(img.shape[0] * y)

                    img = cv2.circle(img, (relative_x,relative_y), 2, (0,0,255), -1)

                    for h_points in horizontal_pts_eyes:
                        left_eye_sym.append(math.hypot(h_points[0] - relative_x, h_points[1] - relative_y))

            count = 0

            cv2.putText(img,'Asymmetries found:',(10,20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)

            if min(left_lip_sym) >2:
                count+=1
                cv2.putText(img,'Left lip',(10,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)

            if min(right_lip_sym) >2:
                if count == 0:
                    cv2.putText(img,'Right lip',(10,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
                elif count == 1:
                    cv2.putText(img,'Right lip',(10,60),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
                count+=1
                
            if min(left_eye_sym) >2:
                if count == 0:
                    cv2.putText(img,'Left eye',(10,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
                elif count == 1:
                    cv2.putText(img,'Left eye',(10,60),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
                elif count == 2:
                    cv2.putText(img,'Left eye',(10,80),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
                count += 1

            if min(right_eye_sym) >2:
                if count == 0:
                    cv2.putText(img,'Right eye',(10,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
                if count == 1:
                    cv2.putText(img,'Right eye',(10,60),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)                
                if count == 2:
                    cv2.putText(img,'Right eye',(10,80),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
                if count == 3:
                    cv2.putText(img,'Right eye',(10,100),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)                


            #Change accuracy
            if min(left_lip_sym) <=2 and min(right_eye_sym) <=2 and min(left_eye_sym)<=2 and min(right_lip_sym)<=2:
                cv2.putText(img,'None',(10,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)

            

        return img

    def head_tilt_detection(self,image):

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False
        
        # Get the result
        results = self.face_mesh.process(image)
        
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, _ = image.shape
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
                if x < -2 or x >2 or y>1 or y<-1:
                    print('Head tilted')
                    return True
                


        return False

    def midpoint(self,p1, p2):
        return (p1[0]+p2[0])/2, (p1[1]+p2[1])/2

    def main_loop(self):
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        while True:
            ret, img = cap.read()
            final_img = img.copy()

            if not ret:
                print('=============Camera not found================')
                break
            

            if not self.head_tilt_detection(img):

                img = self.draw_face_landmarks(img)

                if not self.timer_set:
                    self.timer_set = True
                    self.start_time = time()

                self.end_time = time()

                if self.end_time - self.start_time >=5:
                    final_img = self.get_final_result(final_img)

                    cv2.imwrite('test.jpg',final_img)
                    cv2.destroyAllWindows()
                    cv2.imshow('Final output',final_img)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                
                    return

            else:
                self.start_time = time()
                self.end_time = time()


            cv2.putText(img,'Keep your head straight and smile',(10,20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)

            cv2.imshow('Live feed',img)
            cv2.waitKey(10)

    def main_loop_for_img(self,path):
        
        img = cv2.imread(path)
        

        self.draw_face_landmarks(img)

        cv2.imshow('Live feed',img)
        cv2.waitKey()


objectt = face_summetry()
objectt.main_loop()