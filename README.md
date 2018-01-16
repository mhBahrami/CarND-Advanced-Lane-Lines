# Advanced Lane Finding

In this project, your goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car.

## Table of Content

- [Advanced Lane Finding Project](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#advanced-lane-finding-project)
- [Camera Calibration](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#camera-calibration)
- [Pipeline (single images)](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#pipeline-single-images)
  - [An example of a distortion-corrected image](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#an-example-of-a-distortion-corrected-image)
- [Create a thresholded binary image](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#create-a-thresholded-binary-image)
- [Perspective Transform ](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#perspective-transform)
- [Sliding Window](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#sliding-window)
  - [Locate the Lane Lines and Fit a Polynomial](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#locate-the-lane-lines-and-fit-a-polynomial)
- [Add the Green Zone and Retransform it to the Original Perspective](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#add-the-green-zone-and-retransform-it-to-the-original-perspective)
- [Radius of curvature of the lane and the position of the vehicle with respect to center](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#radius-of-curvature-of-the-lane-and-the-position-of-the-vehicle-with-respect-to-center)
- [`find_pipelines_and_green_zone()`](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#find_pipelines_and_green_zone)
- [Pipeline (video)](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#pipeline-video)
- [Discussion](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#discussion)
  - [Hypothetical Pipeline Failure Cases](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#hypothetical-pipeline-failure-cases)
- [License](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#license)


---

### Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

> **Files**
>
> You can find the code inside [`project.ipynb`](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines/blob/master/project.ipynb) file. Resource images in [`./res`](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines/tree/master/res) folder.

[//]: # "Image References"

[image1]: ./res/undistort_output_calibration05.jpg "Original image (left) and its undistorted version (right)"
[image2]: ./res/undistort_output_test1.jpg "Road Transformed"
[image3]: ./res/test2_sxbinary.jpg "Sobel x Binary"
[image4]: ./res/test2_s_binary.jpg "S Channel Binary"
[image5]: ./res/test2_color_binary.jpg "Color Binary- Combined Version"
[image6]: ./res/test2_source.jpg "Test Image"
[image7]: ./res/test2_region_of_interest.jpg "The Image after Applying the Mask"
[image8]: ./res/test2_region_of_interest_blur.jpg "Blur to Reduce Noise"
[image9]: ./res/test2_region_of_interest_canny.jpg "Canny Edge Detection"
[image10]: ./res/test2_perspective_transform.png "Apply the Perspective Transform"
[image11]: ./res/test2_out_img_1_1.jpg
[image12]: ./res/test2_out_img_1_2.jpg
[image13]: ./res/test2_img_left_plus_hough_lines.jpg "The Starting and Ending Points of Hough Transform Lines for the Left Side"
[image14]: ./res/test2_img_right_plus_hough_lines.jpg "The Starting and Ending Points of Hough Transform Lines for the Right Side"
[image15]: ./res/test2_calculated_curved_pipelines.jpg "The Calculated Pipelines"
[image16]: ./res/test2_curved_pipelines_with_green_zone.jpg "The Calculated Pipelines with Green Zone"
[image17]: ./res/test2_img_trans_rec2org.jpg "Retransform to the Original Perspective"
[image18]: ./res/test2_final1.jpg "Final Result"
[image19]: ./res/off_1.png "Issue with Tuning Parameters by Hand!"
[image20]: ./res/tuning_tool.png "Tuning Tool"
[image21]: ./res/tuned_frame.png "Resolving Issue after Using the Tuning Tool"

---

### Camera Calibration

> The code for this step is contained in the first code cell of the IPython notebook located in the code cells number 2 to 5 in `project.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

You can see other examples here: [[1]](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines/blob/master/res/undistort_output_calibration01.jpg), [[2]](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines/blob/master/res/undistort_output_calibration02.jpg), [[3]](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines/blob/master/res/undistort_output_calibration03.jpg), and [[4]](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines/blob/master/res/undistort_output_calibration04.jpg).

> `cal_undistort(img, objpoints, imgpoints)` in *7th code cell* in `project.ipynb` is responsible to create undistorted image.

### Pipeline (single images)

#### An example of a distortion-corrected image

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

> `cal_undistort(img, objpoints, imgpoints)` in 6th code cell in `project.ipynb` is responsible to create undistorted image.

Steps to create undistorted image:

- Convert the input image to gray

  ```python
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ```

- Calibrate the camera using calculated `objpoints` and `imgpoints`.

  ```python
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  ```

- Undistort the input image using the calibration parameters.

  ```python
  undist = cv2.undistort(img, mtx, dist, None, mtx)
  ```

### Create a thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image. The steps are as following:

- Applying Sobel (x gradient) and S channel color selection after converting the RGB image to HLS.

  > You can see the code in the *13th code cell* of `project.ipynb` (*lines 51 to 98*).

  ```python
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
  sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

  # Threshold color channel
  s_binary = np.zeros_like(s_channel)
  s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
  ```

  |   Sobel x Binary    |  S Channel Binary   |
  | :-----------------: | :-----------------: |
  | ![alt text][image3] | ![alt text][image4] |

- Then combine them:

  ```python
  color_binary[(s_binary == 1) | (sxbinary == 1)] = 1
  ```

  |   Original Image    |  Combined Version   |
  | :-----------------: | :-----------------: |
  | ![alt text][image6] | ![alt text][image5] |

### Perspective Transform 

I used `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` to create an image that includes the pipelines without perspective.

> You can see the code in the *13th code cell* of `project.ipynb` (*lines 1 to 44*).

The result for the sample image is as follows:

| Apply Perspective Transform |
| :-------------------------: |
|    ![alt text][image10]     |

### Sliding Window

#### Locate the Lane Lines and Fit a Polynomial

Steps are as following:

- Take a histogram of the bottom half of the image

  ```python
  histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
  ```

- Create an output image to draw on and  visualize the result.

  ```python
  out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
  ```


- Find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines.

  ```python
  midpoint = np.int(histogram.shape[0]/2)
  leftx_base = np.argmax(histogram[:midpoint])
  rightx_base = np.argmax(histogram[midpoint:]) + midpoint
  ```

- Choose the number of sliding windows and set height of windows

  ```python
  nwindows = 9
  window_height = np.int(binary_warped.shape[0]/nwindows)
  ```

- Identify the x and y positions of all nonzero pixels in the image

  ```python
  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
  ```

   Current positions to be updated for each window

  ```python
  leftx_current = leftx_base
  rightx_current = rightx_base
  ```

- Set the width of the windows +/- margin

  ```python
  margin = 100
  ```

- Set minimum number of pixels found to re-center window

  ```python
  minpix = 50
  ```

- Step through the windows one by one 

  ```python
  for window in range(nwindows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = binary_warped.shape[0] - (window+1)*window_height
      win_y_high = binary_warped.shape[0] - window*window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      # Draw the windows on the visualization image
      cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
      (0,255,0), 2) 
      cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
      (0,255,0), 2) 
      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
      (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
      (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
      # If you found > minpix pixels, recenter next window on their mean position
      if len(good_left_inds) > minpix:
          leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:        
          rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
  ```

- Concatenate the arrays of indices

  ```python
  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)
  ```

- Extract left and right line pixel positions

  ```python
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds] 
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds] 
  ```

- Fit a second order polynomial to each

  ```python
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)
  ```

> You can see the code in the *13th code cell* of `project.ipynb` *lines 101 to 354*.

The results are as following:

| ![alt text][image11] |
| :------------------: |
| ![alt text][image12] |

### Add the Green Zone and Retransform it to the Original Perspective

The safe driving area for a car is the area between the pipelines that I call it "Green Zone." Let's fill the area between red pipelines with the green color using `add_green_zone()` function. For implementing this function I used `cv2.fillPoly()` from OpenCV (figure below). Then retransform it to the original perspective using `transform_back_to_origin()`. 

>  You can see the code in the *13th code cell* of `project.ipynb` *lines 398 to 38 and 33 to 44*.

The results are as following:

| The Calculated Pipelines with Green Zone | Retransform to the Original Perspective |
| :--------------------------------------: | :-------------------------------------: |
|           ![alt text][image16]           |          ![alt text][image17]           |

Now its time to add it to the undistorted frame (below image).

| Add the Green Zone to the Undistorted Frame |
| :--------------------------------------: |
|           ![alt text][image18]           |

### Radius of curvature of the lane and the position of the vehicle with respect to center

I calculated the radius of curvature based on this [tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php). The **radius of curvature** of the curve at a particular point is defined as the radius of the approximating circle. This radius changes as we move along the curve. 

> You can see the code in the*12th code cell* of `project.ipynb` *lines 357 to 399*.

I located the lane line pixels, used their x and y pixel positions to fit a second order polynomial curve:

$f(x) = A y^2 + By +C $  where $y$ in $[a, b]$

**y** is from **a** to **b**. After finding $A$, $B$, and $C$ I calculated the radius of curvature with the following equation:

$R_{curve} = \frac{[1+(2Ay+B)^2]^{3/2}}{|2A|}$

Then I used next equation to calculate position of the center:

$y_{center} = (cols - f_{left}(b) - f_{right}(b))/2 $

if the result value is **negative** it means the vehicle is in the **left** of center and it the result is a **positive** value it means the vehicle is in the **right** of center.

You can see the code here:

```python
def radius_of_curvature(rows, cols, left_fit, right_fit):
  """
     o----o    -                       o--------------o -
    /      \   |    ------------->     |              | |
   /        \  | b                     |              | |d
  o----------o -                       o--------------o - 
  |----a-----|                         |-------c------|

  a(px)=>3.7m                          c(px)=>3.7m
  ym_per_pix=3.7/a                     new_ym_per_pix = 3.7/c
  xm_per_pix=30/b                      new_xm_per_pix = 30/d

  + In Pixel:
  x = f(y) = A*y^2 + B*y + C
  R = [1+(2*A*y+B)^2]^1.5/(2*|A|)  
  + In Meter:
  X = f(Y) = (xm/ym^2)*A*Y^2 + (xm/ym)*B*Y + xm*C
  R = [1 + (2*(xm/ym^2)*A*y+(xm/ym)*B)^2]^1.5/(2*xm*|A|/ym^2)
  """
  # Define conversions in x and y from pixels space to meters
  c, d = cols, rows
  ym_per_pix = 30/d # meters per pixel in y dimension
  xm_per_pix = 3.7/c # meters per pixel in x dimension

  # Generate some fake data to represent lane-line pixels
  ploty = np.linspace(0, rows-1, rows)# to cover same y-range as image
  y_eval = max(ploty)

  # Calculate the new radii of curvature
  Al, Ar = left_fit[0], right_fit[0]
  Bl, Br = left_fit[1], right_fit[1]
  Cl, Cr = left_fit[2], right_fit[2]
  ym, xm = ym_per_pix, xm_per_pix
  left_curverad = (1 + (2*(xm/ym**2)*Al*y_eval + (xm/ym)*Bl)**2)**1.5 / np.absolute(2*xm*Al/ym**2)
  right_curverad = (1 + (2*(xm/ym**2)*Ar*y_eval + (xm/ym)*Br)**2)**1.5 / np.absolute(2*xm*Ar/ym**2)

  # Calculate the vehicle offset from center 
  left_line_pos = Al*y_eval**2 + Bl*y_eval + Cl
  right_line_pos = Ar*y_eval**2 + Br*y_eval + Cr
  pos_to_center = (cols - (left_line_pos+right_line_pos))/2 * xm_per_pix
  _vehicle_pos = 'left' if pos_to_center < 0 else 'right'

  return left_curverad, right_curverad, abs(pos_to_center), _vehicle_pos
```
### `find_pipelines_and_green_zone()`

This function is doing everything for each frame:

> The *14th code cell* in `project.ipynb`.

```python
def find_pipelines_and_green_zone(img, objpoints, imgpoints, prefix = None, _plot = False):
    global left_line, right_line
    use_sliding = left_line==None or right_line==None
    if(left_line==None): left_line = Line()
    if(right_line==None): right_line = Line()
    # Create an undistorted image
    img = cal_undistort(img, objpoints, imgpoints)
    rows,cols,ch = img.shape
    binary_warped = get_binary_warped(img, s_thresh=(170, 255), sx_thresh=(20, 100), _plot=_plot, prefix=prefix)
    if new_mask is not None:
        binary_warped = cv2.bitwise_and(binary_warped, new_mask)
    binary_warped_transf = transform_to_rectangle(binary_warped, rows, cols, prefix=prefix, _plot=_plot)
    left_fit, right_fit = find_pipeline(binary_warped_transf, use_sliding=True, _draw_window=_plot, _plot=_plot, prefix=prefix)
    
    if(left_fit is not None):
        left_line.current_fit = left_fit
    if(right_fit is not None):
        right_line.current_fit = right_fit
    
    if(left_line.current_fit is not None and right_line.current_fit is not None):
        left_fit, right_fit = left_line.current_fit, right_line.current_fit
        left_roc, right_roc, _x_center, _vehicle_pos = radius_of_curvature(rows, cols, left_fit, right_fit)   
        
        img_lines = draw_pred_lines(rows, cols, left_fit, right_fit, line_width=40, prefix=prefix, _plot =_plot)
        img_green_zone = add_green_zone(img_lines, left_fit, right_fit, prefix=prefix, _plot=_plot)
        img_trans_back = transform_back_to_origin(img_green_zone, rows, cols, prefix=prefix, _plot=_plot)
        img_wght = weighted_img(img, img_trans_back)
        final = put_text(img_wght, left_roc, right_roc, _x_center, _vehicle_pos)
    else:
        final = img

    return final
```

### Pipeline (video)

Here's a [link to my video result](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines/tree/master/output) and you can watch it online [here](https://youtu.be/GEifnxlJ8gE).

### Discussion

Sometimes the algorithm is unable to find curved lines, like when the brightness of the road is high. In this case I used the buffered data for the most recent frame which has the valid calculated data. 

Moreover, It was very difficult to tune above parameters. At first I was finding parameters by changing it inside the code. And I spend one day to change them. However, still some frames had problem, like image below.

![alt text][image19]

Finally, I created a tool for tuning with slide bars to change parameters and see the result at the same time. [`tune()`](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines/blob/master/opencv_helper.py#L72) in [`opencv_helper.py`](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines/blob/master/opencv_helper.py) is the function which helps to tune parameters for color selection. I created 14 Trackbar with a value between 0 and 255(below image). There are **2 color selection**, one for yellow and the other for white. Each needs **2 ranges**, a lower range and an upper range. And each range has **3 values**, [H, L, S]. Also **2** Trackbar for canny transform. In total 14 Trackbars (2\*2\*3+2).

![alt text][image20]

And you can see the result after tuning with new method the issue has been resolved.

![alt text][image21]

But it doesn't work properly. So, I used the second method which is **Histogram and Sliding Window** and I described it above. This method produces more accurate results.

#### Hypothetical Pipeline Failure Cases

Pavement fixes and/or combined with other surfaces that create vertical lines near existing road lines.

It would also fail if there was a road crossing or a need to cross lanes or to exit the freeway.

Rain and snow would also have an impact and I’m not sure about night time.

Tail gating a car or a car on a tighter curve would potentially interrupt the visible camera and hence line detection.

### License

[MIT License](LICENSE).
