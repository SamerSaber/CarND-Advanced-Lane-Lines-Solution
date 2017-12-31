import cv2
import camera_calibration
import prespective_transform
import preprocessing
import line_extractor
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

count = 0
def process_frame(frame, DebugImage = False):
    
    
    #undistort the frame
    frame = camera_calibration.undistort(frame, cam_mtx, distortion, DebugImage)
    
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
    left_fitx , right_fitx , ploty , l_curve, r_curve= line_extractor.extract(warped, DebugImage)
    
    #visualize the output
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (frame.shape[1], frame.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    
    ################################################################
    ############# Vehicle offset calculations ######################
    
    camera_position = frame.shape[1]/2
    lane_center = (right_fitx[719] + left_fitx[719])/2
    center_offset = abs(camera_position - lane_center) * (3.7 / (left_fitx[719] - right_fitx[719]))
    print ("center_offset = "+str(center_offset))
    
    ################################################################
    ############# Write the output values ##########################
    message1 = "L_CURVE = "+str(l_curve)
    message2 = "R_CURVE = "+str(r_curve)
    message3 = "Center_Offset = "+str(center_offset)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,message1,(10,500), font, 1, (255,120,120), 1, cv2.LINE_AA)
    cv2.putText(result,message2,(10,550), font, 1, (255,0,120), 1, cv2.LINE_AA)
    cv2.putText(result,message3,(10,600), font, 1, (255,120,0), 1, cv2.LINE_AA)
    global count
    cv2.imwrite("./input/"+str(count)+".png",cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    cv2.imwrite("./output/"+str(count)+".png",cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
    count += 1
    if DebugImage:
        plt.imshow(result)
        plt.show()
    
    return result



if __name__ == '__main__':
    
    static_image = False
    print ("Calibrating the camera....")
    cam_mtx, distortion = camera_calibration.calibrate('./camera_cal')
    if static_image:
        #Calibrate the camera 
        frame = cv2.imread('./input/992.png')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        process_frame(frame, True)
        
    else:
        
    
        white_output = './test_videos_output/project_video.mp4'
        clip1 = VideoFileClip("./project_video.mp4")
#         clip1 = VideoFileClip("./project_video.mp4").subclip(0,10)
        
        #clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
        white_clip = clip1.fl_image(process_frame) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)