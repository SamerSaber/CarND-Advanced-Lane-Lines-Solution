import cv2
import glob
import numpy as np 
import matplotlib.pyplot as plt
def calibrate (path , drawCorners = False):
    calibration_images = sorted(glob.glob(path+'/*.jpg'))
    objPoints = []
    imagePoints = []
    
    objP = np.zeros((9*6,3), np.float32)
    objP[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    for image in calibration_images:
        cv_image = cv2.imread(image)
        gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
        
        ret , corners = cv2.findChessboardCorners(gray , (9,6), None)
        
        if ret == True :
            imagePoints.append(corners)
            objPoints.append(objP)
            
            if drawCorners:
                corners_image = cv2.drawChessboardCorners(cv_image, (9,6), corners, ret)
                cv2.imshow('corners_image',corners_image)
                cv2.waitKey(0)
                
    ret, cam_mtx, distortion, rvecs, tvecs = cv2.calibrateCamera(objPoints, imagePoints, gray.shape[::-1], None, None)
    
    if ret :
        return cam_mtx, distortion
    else:
        return None , None 
        
        
def undistort (image, cam_mtx, distortion, debug_image = False):
    dst = cv2.undistort(image, cam_mtx, distortion, None, cam_mtx)
    
    if debug_image:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('distorted')
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        ax2.set_title('undistorted')
        ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), cmap='gray')
        
        plt.show()
        
       
        
    return dst
    
    
if __name__ == '__main__':
    cam_mtx, distortion = calibrate('./camera_cal', False)
    undistort(cv2.imread('./camera_cal/calibration5.jpg'), cam_mtx, distortion, True)
    