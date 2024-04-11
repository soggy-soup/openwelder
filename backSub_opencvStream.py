import cv2
import numpy as np
import weld_joint
import grbl_gcode
#import util_funcs.displayimg
import time
import copy

import os

#clear old GCODE file
if os.path.isfile("gcode.txt"):
    os.remove("gcode.txt")
    
#loop variable initialization
cont_num = 0
first_frame_counter = 0
cont1 = weld_joint.process_img(None, background=None)
cont2 = weld_joint.process_img(None,background=None)
intersection = None
cont_exist=False

#initialize background object for movement detection
backObject = cv2.createBackgroundSubtractorMOG2(history=400,detectShadows=True)

#set camera parameters   
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2320)#4656 or 2320 for 30fps
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1744)#3496 or 1744 for 30 fps
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
time.sleep(2)
cap.set(cv2.CAP_PROP_FOCUS, 0)

#Initialize video window and UI/UX components
cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.createTrackbar('Focus', 'Live Video', 0, 1023, lambda value: cap.set(cv2.CAP_PROP_FOCUS,value))


#ESC --> destroy window
#A --> start/reset automated capture
#S --> stream GCODE
#R --> Reset to nothing


while True:
    key = cv2.waitKey(1)
    if key == 27:  # Break the loop if 'Esc' key is pressed
        break
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    #pick frame to be background for static contour detection, 
    if first_frame_counter == 10:
        static_background = frame
        corners1 = weld_joint.find_aruco_corners(static_background,camera_matrix=None,dist_coeffs=None)

    #set "a" to reset background and reset contour detection process for automated detection
    if cv2.waitKey(1) & 0xFF == ord('a'):
        static_background = frame
        cont_num = 1
        cont1 = weld_joint.process_img(None, background=None)
        cont2 = weld_joint.process_img(None,background=None)
        intersection = None
        
    #stream gcode 
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if os.path.isfile('gcode.txt'):
            grbl_gcode.stream_gcode('gcode.txt')
        else:
            print("No GCODE to stream")
    
    #'r' to reset to nothing
    if cv2.waitKey(1) & 0xFF == ord('r'):
        cont_num = 0
        cont1 = weld_joint.process_img(None, background=None)
        cont2 = weld_joint.process_img(None,background=None)
        intersection = None     
    
    #for drawing contours
    movement_detected = weld_joint.detect_movement(frame,backObject)
    cv2.putText(frame,f"Movement: {movement_detected}",(100,100),cv2.FONT_HERSHEY_COMPLEX,4, (0,0,255),10,cv2.LINE_AA)
    cv2.drawContours(frame, cont1.contours, -1, (0, 255, 0), thickness = 6)
    cv2.drawContours(frame, cont2.contours, -1, (255, 0, 0), thickness = 6)
    if (intersection is not None):
            cv2_img = weld_joint.draw_intersection(frame, intersection)
        
    
    #Automated contour detection
    if (cont_num>0)&(first_frame_counter > 10):
        path = weld_joint.process_img(frame,background=static_background)
        path.img_bilateral_blur()
        path.back_subtract()
        path.img_detect_GRAY_contours() 
        #path.img_detect_GRAY_contours_adaptive()
        
        if path.hierarchy is not None:
            #print("SOME CONTOURS EXIST")
            path.largest_contour()
            
            if path.contour_area > 5000:
                cont_exist = True
                cv2.drawContours(frame, path.contours, -1, (0, 0, 255), thickness = 10)
                #print("CONTOUR of correct size detected") 
        else:
            cont_exist = False
        
        if  (cont_num==1):
            if (cont_exist is True) & (movement_detected is False):
                print("1st Contour")
                cont_num = 2
                cont1 = copy.deepcopy(path)
                cont2 = weld_joint.process_img(None,background=None)
                intersection = None
                
        elif (cont_num == 2):
            if (cont_exist is True) & (movement_detected is False) & (path.contour_area > 1.25*cont1.contour_area):
                intersection = weld_joint.radius_intersect(cont1.contours,path.contours,radius=25) #and they are intersecting
                if intersection is not None:
                    print("2nd Contour")
                    cont_num = 0
                    cont1 = weld_joint.process_img(None, background=None)
                    cont2 = weld_joint.process_img(None,background=None)
                    intersection_transform = copy.deepcopy(intersection)
                    path_wrt_aruco = weld_joint.transform_points(intersection_transform,corners1,ratio=0.2475)
                    path_wrt_aruco = weld_joint.intersect_cleanup(path_wrt_aruco,e=0.5)
                    grbl_gcode.generate_path_gcode(path_wrt_aruco, 300)
                    

    #dynamically change window size maintaining aspect ratio
    aspect_ratio = frame.shape[1] / frame.shape[0]
    current_width = cv2.getWindowImageRect("Live Video")[2]
    new_height = int((current_width / aspect_ratio))
    cv2.imshow("Live Video", frame)
    cv2.resizeWindow("Live Video", current_width, new_height)  
    
    first_frame_counter += 1
    
cap.release()
cv2.destroyAllWindows()

