import cv2
import numpy as np 
import matplotlib.pyplot as plt
def transform (image , src_pts, dst_pts, Debug_image = False):
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    img_size = (image.shape[1], image.shape[0])  
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_CUBIC)
    reversed = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_CUBIC)
    
    if Debug_image:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('warped')
        ax1.imshow(warped, cmap='gray')
        
        ax2.set_title('reversed')
        ax2.imshow(reversed, cmap='gray')
        plt.show()

        
    return warped, M, Minv
    
    
if __name__ == '__main__':
    
    image = cv2.imread('./fig2.jpg')
    src_pts = np.float32(
                        [[180  , 665],
                         [1111 , 655],
                         [562  , 460],
                         [726  , 460]])
    dst_pts = np.float32(
                        [[320  , 640],
                         [1000 , 640],
                         [320  , 150],
                         [1000 , 150]])
    
    transform(image,src_pts,dst_pts,True)
    plt.imshow(image)
    plt . plot(180,665,'.')
    plt . plot(1111,655,'.')
    plt . plot(562,460,'.')
    plt . plot(726,460,'.')
    plt.show()