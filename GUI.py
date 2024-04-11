from tkinter import *
from PIL import Image, ImageTk
import cv2
import customtkinter
import numpy as np
import weld_joint
import grbl_gcode
import copy
import time
import os

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        #initialize window properties
        self.bind('<Escape>', lambda e: self.quit())
        self.title("CNC Welder")
        self.resizable(width=False, height=False)
        customtkinter.set_appearance_mode("dark")
        
        #initialize contours as empty
        self.contours1 = None
        self.contours2 = None
        self.joint_intersection = None
        self.path_wrt_aruco = None
        self.corners = None
        
        #switch states to ensure first picture is taken before second
        self.img1_state = NORMAL
        self.img2_state = DISABLED
        self.intersection_state = DISABLED
        self.GCODE_state = DISABLED
        
        self.frame_counter = 0
        self.tf_contours = True
        
        #setup video stream with opencv      
        video_width, video_height = 4656,3496
        video_aspect_ratio = video_width/video_height
        tot_rows = 15
        self.cap = cv2.VideoCapture(0,cv2.CAP_MSMF)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2320)#4656 or 2320 for 30fps
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1744)#3496 or 1744 for 30 fps
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(2)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0) #set camera focus
        self.video_win_width, self.video_win_height = 800, int(800/video_aspect_ratio)     
        
        #create video stream label
        self.video_stream = customtkinter.CTkLabel(self,text="")
        self.video_stream.grid(row=0,column=0, padx=10,pady=10,rowspan=tot_rows)
        
        #frame for all button controls
        right_frame = customtkinter.CTkFrame(self, width=600, height=self.video_win_height)
        right_frame.grid(row=0,column=1,padx=10,pady=10,columnspan=3,rowspan=tot_rows)


        #image processing buttons
        customtkinter.CTkLabel(self, text="Joint Detection:").grid(row=1,column=1)
        self.reset_background = customtkinter.CTkButton(self,text="Resest Background",command = self.generate_background).grid(row=2,column=1)
        self.img1 = customtkinter.CTkButton(self,text="Picture 1",fg_color="red",state=self.img1_state, command=self.process_img1).grid(row=3,column=1)
        self.img2 = customtkinter.CTkButton(self,text="Picture 2",fg_color="blue",state=self.img2_state,command=self.process_img2).grid(row=3,column=2)
        self.joint_location = customtkinter.CTkButton(self,text="Weld Joint",state=self.intersection_state,command = self.joint).grid(row=3,column=3)
        self.generate_gcode = customtkinter.CTkButton(self,text="GCODE",state=self.GCODE_state,command = self.gcode).grid(row=4,column=1)
        self.stream_gcode = customtkinter.CTkButton(self,text="Stream",state=NORMAL,command = self.strm).grid(row=4,column=2)
        self.stop_stream = customtkinter.CTkButton(self,text="STOP: ctr + x",state=None,command = None).grid(row=4,column=3)

        self.open_camera()
        
    def open_camera(self):
        _, self.frame = self.cap.read()
        
        if (self.frame_counter > 20) & (self.tf_contours == True):
            self.path = weld_joint.process_img(None,background=self.background)
            self.path.img_in_processing = self.frame
            self.path.img_bilateral_blur()
            self.path.back_subtract()
            self.path.img_detect_GRAY_contours() 
            if self.path.hierarchy is not None:
                self.path.largest_contour()
                cv2_img = cv2.drawContours(self.frame, self.path.contours, -1, (0,255,255), thickness = 7)
            
        cv2_img = cv2.drawContours(self.frame, self.contours1, -1, (0,0,255), thickness = 7)
        cv2_img = cv2.drawContours(self.frame, self.contours2, -1, (0,255,0), thickness = 7)
        
        if (self.joint_intersection is not None):
            cv2_img = weld_joint.draw_intersection(self.frame, self.joint_intersection)
        
        cv2_img =  cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGBA)
        cap_img = Image.fromarray(cv2_img)
        photo_img = customtkinter.CTkImage(light_image=cap_img,size=(self.video_win_width,self.video_win_height))
        self.video_stream.photo_img = photo_img
        self.video_stream.configure(image =photo_img)
        self.video_stream.after(10,self.open_camera)
        self.frame_counter += 1
        if self.frame_counter == 10:
            self.background = self.frame
            self.corners = weld_joint.find_aruco_corners(self.frame)
            
    def process_img1(self):
        if self.path.hierarchy is not None:               
            self.contours2 = None
            self.joint_intersection = None
            self.contours1 = copy.deepcopy(self.path.contours)
            
            self.img2_state = NORMAL
            self.grid_slaves(row=3, column=2)[0].configure(state=self.img2_state)
            self.intersection_state = DISABLED
            self.grid_slaves(row=3, column=3)[0].configure(state=self.intersection_state)
            self.GCODE_state = DISABLED
            self.grid_slaves(row=4, column=1)[0].configure(state=self.GCODE_state)
        else: 
            print("No Contours")
            
    def process_img2(self):
        if self.path.hierarchy is not None:
            self.joint_intersection = None
            self.contours2 = copy.deepcopy(self.path.contours)
            
            self.img2_state = DISABLED
            self.grid_slaves(row=3, column=2)[0].configure(state=self.img2_state)
            self.GCODE_state = DISABLED
            self.grid_slaves(row=4, column=1)[0].configure(state=self.GCODE_state)
            self.intersection_state = NORMAL
            self.grid_slaves(row=3, column=3)[0].configure(state=self.intersection_state)
        else: 
            print("No Contours")
            
    def generate_background(self):
        #self.background == self.frame
        self.frame_counter = 0
        self.tf_contours = True
        self.contours1 = None
        self.contours2 = None
        self.joint_intersection = None
        self.intersection_state = DISABLED
        self.grid_slaves(row=3, column=3)[0].configure(state=self.intersection_state)
        self.GCODE_state = DISABLED
        self.grid_slaves(row=4, column=1)[0].configure(state=self.GCODE_state)
        self.img2_state = DISABLED
        self.grid_slaves(row=3, column=2)[0].configure(state=self.img2_state)
        self.GCODE_state = DISABLED
        self.grid_slaves(row=4, column=1)[0].configure(state=self.GCODE_state)
        print("background reset")
    
    def joint(self):
        if ((self.contours1 is not None) & (self.contours2 is not None) & (self.corners is not None)):
            self.joint_intersection = weld_joint.radius_intersect(self.contours1,self.contours2,radius=35)
            if (self.joint_intersection is not None) & (self.corners != ()):
                self.intersection_state = DISABLED
                self.grid_slaves(row=3, column=3)[0].configure(state=self.intersection_state)
                self.GCODE_state = NORMAL
                self.grid_slaves(row=4, column=1)[0].configure(state=self.GCODE_state)
                self.contours1 = None
                self.contours2 = None
                inter = copy.deepcopy(self.joint_intersection)
                pxmmratio = weld_joint.mm_to_px_ratio(self.corners, aruco_size_mm=50)
                print(pxmmratio)
                self.path_wrt_aruco = weld_joint.transform_points(inter,self.corners,ratio=(.248))
                self.path_wrt_aruco = weld_joint.intersect_cleanup(self.path_wrt_aruco,e=0.45)
                self.tf_contours = False
            else:
                print("No ARUCO TAG")
            
                              
    def gcode(self):
        if (self.path_wrt_aruco is not None):
            grbl_gcode.generate_path_gcode(self.path_wrt_aruco, 300)
    
    def strm(self):
        if os.path.isfile('gcode.txt'):
            grbl_gcode.stream_gcode('gcode.txt')
        else:
            print("No GCODE to stream")
         
app = App()
app.mainloop()

