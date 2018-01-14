# Advanced Lane Finding

In this project, your goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car.

## Table of Content

- [Advanced Lane Finding Project](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#advanced-lane-finding-project)
- [Camera Calibration](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#camera-calibration)
- [Pipeline (single images)](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#pipeline-single-images)
  - [An example of a distortion-corrected image](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#an-example-of-a-distortion-corrected-image)
- [Create a thresholded binary image and apply canny transform](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#create-a-thresholded-binary-image-and-apply-canny-transform)
- [Perspective Transform and Create Images for the Left and Right Lines](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#perspective-transform-and-create-images-for-the-left-and-right-lines)
- [Hough Transform](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#hough-transform)
- [Find the Best Line with Polynomial Interpolation](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#find-the-best-line-with-polynomial-interpolation)
- [Add the Green Zone and Retransform it to the Original Perspective](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#add-the-green-zone-and-retransform-it-to-the-original-perspective)
- [Radius of curvature of the lane and the position of the vehicle with respect to center](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#radius-of-curvature-of-the-lane-and-the-position-of-the-vehicle-with-respect-to-center)
- [Tuning the Parameters](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#tuning-the-parameters)
- [Pipeline (video)](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#pipeline-video)
- [Discussion](https://github.com/mhBahrami/CarND-Advanced-Lane-Lines#discussion)
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
[image3]: ./res/test2_mask_y.jpg "Generated Mask for Yellow Color"
[image4]: ./res/test2_mask_w.jpg "Generated Mask for White Color"
[image5]: ./res/test2_mask_wy.jpg "Generated Mask for Yellow & White Colors"
[image6]: ./res/test2_source.jpg "Test Image"
[image7]: ./res/test2_region_of_interest.jpg "The Image after Applying the Mask"
[image8]: ./res/test2_region_of_interest_blur.jpg "Blur to Reduce Noise"
[image9]: ./res/test2_region_of_interest_canny.jpg "Canny Edge Detection"
[image10]: ./res/test2_img_trans_org2rec.jpg "Apply the Perspective Transform"
[image11]: ./res/test2_img_left.jpg "The Left Pipeline Image"
[image12]: ./res/test2_img_right.jpg "The Right Pipeline Image"
[image13]: ./res/test2_img_left_plus_hough_lines.jpg "The Starting and Ending Points of Hough Transform Lines for the Left Side"
[image14]: ./res/test2_img_right_plus_hough_lines.jpg "The Starting and Ending Points of Hough Transform Lines for the Right Side"
[image15]: ./res/test2_calculated_curved_pipelines.jpg "The Calculated Pipelines"
[image16]: ./res/test2_curved_pipelines_with_green_zone.jpg "The Calculated Pipelines with Green Zone"
[image17]: ./res/test2_img_trans_rec2org.jpg "Retransform to the Original Perspective"
[image18]: ./res/test2_final.jpg "Add the Green Zone to the Undistorted Frame"

---

### Camera Calibration

> The code for this step is contained in the first code cell of the IPython notebook located in the code cells number 2 to 4 in `project.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

You can see other examples here, here, hand here.

> `cal_undistort(img, objpoints, imgpoints)` in 6th code cell in `project.ipynb` is responsible to create undistorted image.

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

### Create a thresholded binary image and apply canny transform

I used a combination of color and gradient thresholds to generate a binary image. Then I used the canny edge detection method to create an image for finding the lines.

The steps are as following:

- Create a mask to select white and yellow color regions in the image using `get_white_yellow_mask()` function. I created masks for white and yellow colors using `cv2.inRange()` separately and then I combined them using `cv2.bitwise_and()`. you can see the result below.

  > You can see the code in the*12th code cell* of `project.ipynb` (*lines 65 to 112*).

  | Generated Mask for Yellow Color | Generated Mask for White Color |
  | :-----------------------------: | :----------------------------: |
  |       ![alt text][image3]       |      ![alt text][image4]       |

  |   Original Image    | Generated Mask for Yellow and White colors |
  | :-----------------: | :--------------------------------------: |
  | ![alt text][image6] |           ![alt text][image5]            |

- Apply the mask to the original image. Then, make it blur to reduce the noise using `cv2.GaussianBlur()`. Apply the **canny transform** to the blurred image.

  > You can see the code in the*12th code cell* of `project.ipynb` (*lines 155 to 179*).

  |   Apply the Mask    |   Blur the Image    |
  | :-----------------: | :-----------------: |
  | ![alt text][image7] | ![alt text][image8] |

  | Apply the Canny Transform |
  | :-----------------------: |
  |    ![alt text][image9]    |


### Perspective Transform and Create Images for the Left and Right Lines 

I used `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` to create an image that includes the pipelines without perspective. Before finding the left and right curved pipelines, I separated the left and right lines using `separate_to_left_right()` function.

> You can see the code in the*12th code cell* of `project.ipynb` (*lines 182 to 248*).

The result for the sample image is as follows:

| Apply Perspective Transform |
| :-------------------------: |
|    ![alt text][image10]     |

| The Left Pipeline Image | The Right Pipeline Image |
| :---------------------: | :----------------------: |
|  ![alt text][image11]   |   ![alt text][image12]   |

### Hough Transform

Next step is applying _Hough Transform_ by using `find_hough_lines()` function to the both left and right side images. The results for each side is a set of lines and each line contains a couple of points that indicates the starting and ending points of it.

> You can see the code in the*12th code cell* of `project.ipynb` (*lines 251 to 278*).

The result is as following:

| The Starting and Ending Points of Hough Transform Lines for the Left Side | The Starting and Ending Points of Hough Transform Lines for the Right Side |
| :--------------------------------------: | :--------------------------------------: |
|           ![alt text][image13]           |           ![alt text][image14]           |

### Find the Best Line with Polynomial Interpolation

`scikit-learn` has some pakcages for Polynomial Interpolation (see an example [here](http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html)). In fact, I used linear regression with polynomial features to approximate a nonlinear function that fits the points which were found from the previous step. I used the following packages:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
```

The function that is responsible to fit the best line is `find_pipeline()`. The polynomial with a degree equal to 2 gives the best results. If you increase or decrease the degree of polynomial, the underfitting and overfitting problems occur. For more information see [here](http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py). Finally, the calculated pipeline formula would be used to draw pipelines using `draw_pred_lines()` function (image below).

> You can see the code in the*12th code cell* of `project.ipynb` *lines 281 to 423*.

| The Calculated Pipelines |
| :----------------------: |
|   ![alt text][image15]   |

### Add the Green Zone and Retransform it to the Original Perspective

The safe driving area for a car is the area between the pipelines that I call it "Green Zone." Let's fill the area between red pipelines with the green color using `add_green_zone()` function. For implementing this function I used `cv2.fillPoly()` from OpenCV (figure below). Then retransform it to the original perspective using `transform_back_to_origin()`. 

>  You can see the code in the*12th code cell* of `project.ipynb` *lines 214 to 225 and 426 to 440*.

The results are as following:

| The Calculated Pipelines with Green Zone | Retransform to the Original Perspective |
| :--------------------------------------: | :-------------------------------------: |
|           ![alt text][image16]           |          ![alt text][image17]           |

Now its time to add it to the undistorted frame (below image).

| Add the Green Zone to the Undistorted Frame |
| :--------------------------------------: |
|           ![alt text][image18]           |

### Radius of curvature of the lane and the position of the vehicle with respect to center

I calculated the radius of curvature based on this [tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php). The **radius of curvature** of the curve at a particular point is defined as the radius of the approximating circle. This radius changes as we move along the curve. $ 

> You can see the code in the*12th code cell* of `project.ipynb` *lines 443 to 463*.

I located the lane line pixels, used their x and y pixel positions to fit a second order polynomial curve:

$f(x) = A y^2 + By +C$

After finding $A$, $B$, and $C$ I calculated the radius of curvature with the following equation:

$R_{curve} = \frac{[1+(2Ay+B)^2]^{3/2}}{|2A|}$

Then I used next equation to calculate position of the center:

$y_{center} = - \frac{R_{curve}}{\sqrt{1+m^2}}\times\frac{A}{|A|}+y_{vehicle} $

where $m$ is the slope of the **normal**. You can find more info [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) in *"Example 1."*

### Tuning the Parameters

I put the parameters for different parts of this code in one place to tune them conveniently. The final result for these parameters are as following:

```python
# Color in range:project_video 
y_lower = [10, 0, 120]
y_upper = [40, 255, 255]
w_lower = [16, 182, 0]
w_upper = [255, 255, 255]

# Transform:project_video
coef_w_top = 0.15
coef_w_dwn = 1.00
offset_v_top = 0.63
offset_v_dwn = 0.05

# Blur
kernel_size = 9

# Canny
canny_low_threshold = 0
canny_high_threshold = 255

# Make a blank the same size as our image to draw on
rho = 1                 # distance resolution in pixels of the Hough grid
theta = np.pi/180 * 0.5 # angular resolution in radians of the Hough grid
threshold = 10          # minimum number of votes (intersections in Hough grid cell)
min_line_len = 15       # minimum number of pixels making up a line
max_line_gap = 10       # maximum gap in pixels between connectable line segments

# Interpolation
degree = 2

# plot 
lw = 2
```

### Pipeline (video)

Here's a [link to my video result](./project_video.mp4) and you can watch it online [here](https://youtu.be/EqNMOwQrt-w).

### Discussion

Sometimes the algorithm is unable to find curved lines, like when the brightness of the road is high. In this case I used the buffered data for the most recent frame which has the valid calculated data. 

### License

[MIT License](LICENSE).
