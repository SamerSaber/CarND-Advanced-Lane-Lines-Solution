import cv2
import camera_calibration
import prespective_transform
import preprocessing
import line_extractor
import matplotlib.pyplot as plt
import numpy as np
def process_undistorted_frame(frame, DebugImage = False):
    
    
    bin_image = preprocessing.process(frame, DebugImage)
    #prespective transformation
    src_pts = np.float32(
                        [[180  , 665],
                         [1111 , 655],
                         [562  , 460],
                         [726  , 460]])
    dst_pts = np.float32(
                        [[320  , 700],
                         [1000 , 700],
                         [320  , 150],
                         [1000 , 150]])
     
    warped , M, Minv = prespective_transform.transform(bin_image, src_pts, dst_pts, DebugImage)
    
    #Extract the lane lines
    line_extractor.extract(warped, DebugImage)
    
#     if DebugImage:
#         
#         
#         f, (ax1, ax2 , ax3) = plt.subplots(1, 3, figsize=(20,10))
#         ax1.set_title('mask')
#         ax1.imshow(mask, cmap='gray')
#         
#         ax2.set_title('bin_image')
#         ax2.imshow(bin_image, cmap='gray')
#         
#         ax3.set_title('warped')
#         ax3.imshow(warped, cmap='gray')
#         plt.savefig("warped.jpg")
#         plt.show()
    
    
    return



if __name__ == '__main__':
    #Calibrate the camera 
    cam_mtx, distortion = camera_calibration.calibrate('./camera_cal')
    
    #undistort the frame
    frame = camera_calibration.undistort(cv2.imread('./test_images/test3.jpg'), cam_mtx, distortion)
    process_undistorted_frame(frame, True)