import cv2
import numpy as np
from rdp import rdp

class process_img:
    def __init__(self, img_path,background=None):
        self.img_path = img_path
        self.img_in_processing = self.img_path 
        self.contours = []
        self.hierarchy = []
        self.gray = []
        self.saturate = []
        self.thresh = []
        self.contour_area = []
        self.background = background

    def img_crop(self, h_start,h_end, w_start,w_end):
        self.h_start = h_start
        self.h_end = h_end
        self.w_start = w_start
        self.w_end = w_end
        self.crop_height = slice(h_start,h_end)
        self.crop_width = slice(w_start,w_end)
        self.img_in_processing = self.img_in_processing[self.crop_height, self.crop_width]

    def img_gaussian_blur(self):
        self.img_in_processing = cv2.GaussianBlur(self.img_in_processing, (21, 21), 0)

    def img_median_blur(self):
        self.img_in_processing = cv2.medianBlur(self.img_in_processing, 5)

    def img_bilateral_blur(self):
        self.img_in_processing = cv2.bilateralFilter(self.img_in_processing, 15, 175, 175)

    def img_basic_blur(self):
        self.img_in_processing = cv2.blur(self.img_in_processing, (100, 100))

    def back_subtract(self):
        self.img_in_processing = cv2.absdiff(self.background,self.img_in_processing)
        
    def img_detect_GRAY_contours(self):
        self.gray = cv2.cvtColor(self.img_in_processing, cv2.COLOR_RGB2GRAY)
        _, self.thresh = cv2.threshold(self.gray, 10, 255, cv2.THRESH_BINARY)
        self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
        self.contours, self.hierarchy = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    def img_detect_GRAY_contours_adaptive(self):
        self.gray = cv2.cvtColor(self.img_in_processing, cv2.COLOR_RGB2GRAY)
        self.thresh = cv2. adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11,2)
        self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
        self.contours, self.hierarchy = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    def img_detect_HSV_contours(self):
        self.saturate = cv2.cvtColor(self.img_in_processing, cv2.COLOR_RGB2HSV)
        self.thresh = cv2.inRange(self.saturate, (0,50,0), (50,255,200))
        self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        self.contours, self.hierarchy = cv2.findContours(self.thresh[self.crop_height, self.crop_width], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(self.w_start, self.h_start))
        
    def img_draw_contours(self):
        self.img_in_processing = cv2.drawContours(self.img_read, self.contours, -1, (0, 0, 255), thickness = 10)
        
    def largest_contour(self):
        for i in range(len(self.hierarchy[0])):
            areaN = cv2.contourArea(self.contours[i])
            self.contour_area.append(areaN)
        self.contour_area = np.array(self.contour_area)
        max_area_idx = np.argmax(self.contour_area)
        self.contour_area = self.contour_area[max_area_idx]
        self.contours = self.contours[max_area_idx]       
        
            
def radius_intersect(cont1, cont2, radius = None):
    #radial tolerance for what is considered an "intersection", squared to reduce calcs below
    rad_squared = radius ** 2
    
    #resizing arrays to make math easier
    cont1 = cont1.squeeze()
    cont2 = cont2.squeeze()
        
    #find deltaX/deltaY and calculate distance between all points
    deltas = cont1[:,None,:] - cont2[None,:,:]
    distances = np.sum(np.square(deltas), axis = -1)
       
    #find index of coordinates that are within the overlap range
    overlap = np.flatnonzero(distances <= rad_squared)
    idx_delete = np.unravel_index(overlap,distances.shape)
    
    #delete any points that are within the overlap range
    cont1_delete = np.delete(cont1, idx_delete[0], 0)
    #cont2 = np.delete(cont2, idx_delete[1], 0)
    
    #resize contours to 4D array to plot with openCV
    cont1_delete = cont1_delete[None,:,None]
    #cont2 = cont2[None,:,None]
        
    #contour containing "weld joint"
    if len(idx_delete[0]) < 1000:
        #print("NO INTERSECTION")
        intersection_contour = None
    else: 
        intersection_contour = cont1_delete

    return intersection_contour

def find_aruco_corners(img,camera_matrix=None,dist_coeffs=None):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    if ((camera_matrix is not None) & (dist_coeffs is not None)):
        img = cv2.undistort(img, camera_matrix, dist_coeffs) 
    params = cv2.aruco.DetectorParameters()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters= params)
    return marker_corners
    
def transform_points(path, corners,ratio=None):
    x1,y1 = corners[0][0][0]
    x2,y2 = corners[0][0][3]
    theta = np.arctan((y1-y2)/(x1-x2)) 
    print(x1)
    print(y1)
    print(x2)
    print(y2)
    path[:,:,:,0] = ((path[:,:,:,0]-x1)*np.cos(theta))+((y1-path[:,:,:,1])*np.sin(theta))
    path[:,:,:,1] = (-(path[:,:,:,0]-x1)*np.sin(theta))+((y1-path[:,:,:,1])*np.cos(theta))
    transformed_path = np.round(ratio*path,2)
    return transformed_path

def intersect_cleanup(intersection_contour,e=None):
    intersection_contour = intersection_contour.squeeze()
    cleaned = rdp(intersection_contour,epsilon=e,algo="rec")
    cleaned = cleaned[None,:,None]
    return cleaned
 
def detect_movement(frame,background_object):
    fgmask = background_object.apply(frame,learningRate=-1)
    _, fgmask = cv2.threshold(fgmask, 250,255, cv2.THRESH_BINARY) #remove shaddows
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    movmtCont, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    movement = False
    for i in movmtCont:
        if cv2.contourArea(i) > 65:
            movement = True
            
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    
    return movement
 
def draw_intersection(frame,intersection_contour):
    for i in range(len(intersection_contour[0])):
        image = cv2.circle(frame, (intersection_contour[0][i][0][0],intersection_contour[0][i][0][1]), radius=3, color=(255,255,255), thickness=-1)
    return image
    
#Auto px to real world mm Calibration function isn't that great, manual calibration with a ruler works better (then double check by actually moving the machine)
def mm_to_px_ratio(corners, aruco_size_mm=None):
    avg_dist = 0
    for i in range(len(corners[0][0])):
        avg_dist += np.sqrt(np.square(corners[0][0][i-1][0]-corners[0][0][i][0]) + np.square(corners[0][0][i-1][1]-corners[0][0][i][1]))
    ratio = (aruco_size_mm*4)/avg_dist
    return ratio















