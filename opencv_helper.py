import cv2
import numpy as np
import matplotlib.image as mpimg
import pickle

def nothing(x):
    pass

def get_binary_warped(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255
        
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255
        
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    color_binary = np.zeros_like(sxbinary)
    color_binary[(s_binary == 255) | (sxbinary == 255)] = 255
    
    return color_binary, sxbinary, s_binary


def read_data(file_name):    
    with open(file_name, mode='rb') as f:
        data = pickle.load(f)
        
    return data['objpoints'], data['imgpoints']


def cal_undistort(img, objpoints, imgpoints):
    """
    - takes an image(RGB), object points, and image points,
    - performs the camera calibration, image distortion correction, and 
    - returns the undistorted image
    """
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def get_white_yellow_mask_h(img,y_lower,y_upper,w_lower,w_upper):
    # Convert RGB to HLS
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # Yellow mask
    # define range of yellow color in HLS color space
    y_lower_range = np.array(y_lower, dtype=np.uint8)
    y_upper_range = np.array(y_upper, dtype=np.uint8)
    # Apply the range values to the HLS image to get only yellow colors
    mask_y = cv2.inRange(img_hls, y_lower_range, y_upper_range)

    # White mask
    # define range of white color in HLS color space
    w_lower_range = np.array(w_lower, dtype=np.uint8)
    w_upper_range = np.array(w_upper, dtype=np.uint8)
    # Apply the range values to the HLS image to get only white colors
    mask_w = cv2.inRange(img_hls, w_lower_range, w_upper_range)
    
    mask_wy = cv2.add(mask_y, mask_w)
    
    mask = np.zeros_like(img)
    mask[:,:,0] = mask_wy
    mask[:,:,1] = mask_wy
    mask[:,:,2] = mask_wy
    
    return np.copy(mask)

def get_canny(img, mask, canny_low_threshold, canny_high_threshold, kernel_size):
    region_of_interest = img if mask is None else cv2.bitwise_and(img, mask)
    region_of_interest_blur = gaussian_blur(region_of_interest, kernel_size)
    region_of_interest_canny = canny(region_of_interest_blur, canny_low_threshold, canny_high_threshold)
    
    return np.copy(region_of_interest_canny)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def test(img1):
    # Create a black image, a window
    # create trackbars for color change
#     cv2.createTrackbar('YH(Low)','conrols',18,255,nothing)#8
#     cv2.createTrackbar('YH(high)','conrols',255,255,nothing)
#     cv2.createTrackbar('YL(Low)','conrols',184,255,nothing)#190
#     cv2.createTrackbar('YL(high)','conrols',255,255,nothing)
#     cv2.createTrackbar('YS(Low)','conrols',100,255,nothing)#18
#     cv2.createTrackbar('YS(high)','conrols',255,255,nothing)
    
#     cv2.createTrackbar('WH(Low)','conrols',0,255,nothing)
#     cv2.createTrackbar('WH(high)','conrols',106,255,nothing)
#     cv2.createTrackbar('WL(Low)','conrols',50,255,nothing)#16
#     cv2.createTrackbar('WL(high)','conrols',255,255,nothing)#180
#     cv2.createTrackbar('WS(Low)','conrols',100,255,nothing)
#     cv2.createTrackbar('WS(high)','conrols',254,255,nothing)
    
#     cv2.createTrackbar('YH(Low)','conrols',8,255,nothing)
#     cv2.createTrackbar('YH(high)','conrols',255,255,nothing)
#     cv2.createTrackbar('YL(Low)','conrols',194,255,nothing)
#     cv2.createTrackbar('YL(high)','conrols',255,255,nothing)
#     cv2.createTrackbar('YS(Low)','conrols',0,255,nothing)
#     cv2.createTrackbar('YS(high)','conrols',255,255,nothing)
    
#     cv2.createTrackbar('WH(Low)','conrols',0,255,nothing)
#     cv2.createTrackbar('WH(high)','conrols',106,255,nothing)
#     cv2.createTrackbar('WL(Low)','conrols',16,255,nothing)
#     cv2.createTrackbar('WL(high)','conrols',180,255,nothing)
#     cv2.createTrackbar('WS(Low)','conrols',84,255,nothing)
#     cv2.createTrackbar('WS(high)','conrols',254,255,nothing)
    
#     cv2.createTrackbar('Canny(Low)','conrols',21,255,nothing)
#     cv2.createTrackbar('Canny(high)','conrols',143,255,nothing)
    
    cv2.namedWindow('conrols', cv2.WINDOW_NORMAL)
    cv2.namedWindow('color_binary', cv2.WINDOW_NORMAL)
    cv2.namedWindow('sxbinary', cv2.WINDOW_NORMAL)
    cv2.namedWindow('s_binary', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('s_thresh(Low)','conrols',100,255,nothing)
    cv2.createTrackbar('s_thresh(high)','conrols',255,255,nothing)
    cv2.createTrackbar('sx_thresh(Low)','conrols',8,255,nothing)
    cv2.createTrackbar('sx_thresh(high)','conrols',196,255,nothing)

    namelist = ['straight_lines1.jpg',
                 'straight_lines2.jpg',
                 'test1.jpg',
                 'test2.jpg',
                 'test3.jpg',
                 'test4.jpg',
                 'test5.jpg',
                 'test6.jpg',
                 'test7.jpg']
    kernel_size = 9
    idx = 7
    path = './test_images/{0}'.format(namelist[idx])
    file_name = 'chessboard_corners.p'

    #img1 = mpimg.imread(path)
    objpoints, imgpoints = read_data(file_name)
    img = cal_undistort(img1, objpoints, imgpoints)

#     mask = np.zeros_like(img)
#     result = np.zeros_like(img)
    color_binary = np.zeros_like(img[:,:,0])
    sxbinary = np.zeros_like(img[:,:,0])
    s_binary = np.zeros_like(img[:,:,0])

    while(1):
        cv2.imshow('color_binary',color_binary)
        cv2.imshow('s_binary',s_binary)
        cv2.imshow('sxbinary',sxbinary)
        k = cv2.waitKey(1)
        if k == 27:
            break

        # get current positions of four trackbars
        s_threshl = cv2.getTrackbarPos('s_thresh(Low)','conrols')
        s_threshh = cv2.getTrackbarPos('s_thresh(high)','conrols')
        sx_threshl = cv2.getTrackbarPos('sx_thresh(Low)','conrols')
        sx_threshh = cv2.getTrackbarPos('sx_thresh(high)','conrols')
        s_thresh0 = (s_threshl, s_threshh)
        sx_thresh0 = (sx_threshl, sx_threshh)
        
        color_binary, sxbinary, s_binary = get_binary_warped(img, s_thresh=s_thresh0, sx_thresh=sx_thresh0)

#         # get current positions of four trackbars
#         y_hl = cv2.getTrackbarPos('YH(Low)','conrols')
#         y_hh = cv2.getTrackbarPos('YH(high)','conrols')
#         y_ll = cv2.getTrackbarPos('YL(Low)','conrols')
#         y_lh = cv2.getTrackbarPos('YL(high)','conrols')
#         y_sl = cv2.getTrackbarPos('YS(Low)','conrols')
#         y_sh = cv2.getTrackbarPos('YS(high)','conrols')
#         y_lower = np.array([y_hl, y_ll, y_sl])
#         y_upper = np.array([y_hh, y_lh, y_sh])

#         w_hl = cv2.getTrackbarPos('WH(Low)','conrols')
#         w_hh = cv2.getTrackbarPos('WH(high)','conrols')get_binary_warped
#         w_ll = cv2.getTrackbarPos('WL(Low)','conrols')
#         w_lh = cv2.getTrackbarPos('WL(high)','conrols')
#         w_sl = cv2.getTrackbarPos('WS(Low)','conrols')
#         w_sh = cv2.getTrackbarPos('WS(high)','conrols')
#         w_lower = np.array([w_hl, w_ll, w_sl])
#         w_upper = np.array([w_hh, w_lh, w_sh])

#         cl = cv2.getTrackbarPos('Canny(Low)','conrols')
#         ch = cv2.getTrackbarPos('Canny(high)','conrols')

#         mask = get_white_yellow_mask_h(img,y_lower,y_upper,w_lower,w_upper)

#         result = get_canny(img, mask, cl, ch, kernel_size)

    cv2.destroyAllWindows()
