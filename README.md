# Camera Pose Estimation using Homography

**Note** : This project was done as a part of the course ENPM673 - Perception for Autonomous Robots, in Spring 2023 at the University of Maryland, College Park.

## Overview
This project addresses the task of estimating the pose of a camera observing a planar object using homography techniques. The pipeline involves various steps including edge detection, corner extraction, homography computation, and camera pose estimation.

## Pipeline
1. **Edge Detection:** Utilize the Canny edge detector to identify edges in the camera frames.

![image](https://github.com/Shyam-pi/Camera-Pose-Estimation-using-Homography/assets/57116285/2d5057ce-68d4-4c47-aa9c-e0a966741e58)

2. **Corner Extraction:** Employ the Hough Transform to extract the corners of the paper from the detected edges. Compute equations of the 4 edges followed by computing their points of intersection

![image](https://github.com/Shyam-pi/Camera-Pose-Estimation-using-Homography/assets/57116285/4e50087b-4cf3-4a4d-b2c6-cfe898e8a3fd)

![image](https://github.com/Shyam-pi/Camera-Pose-Estimation-using-Homography/assets/57116285/8c7389f9-7a70-4d89-8cee-86cc89cc3c5b)

3. **Homography Computation:** Compute the homography matrix to map pixel coordinates to world coordinates of the paper.
4. **Camera Pose Estimation:** Decompose the homography matrix to retrieve the rotation and translation vectors of the camera.
5. **Plotting Results:** Visualize the camera's roll, pitch, yaw, and translations over frames.

![image](https://github.com/Shyam-pi/Camera-Pose-Estimation-using-Homography/assets/57116285/2aa22a3f-1261-4b78-a8be-07cad23bacfa)

![image](https://github.com/Shyam-pi/Camera-Pose-Estimation-using-Homography/assets/57116285/528309b1-82d1-41d7-9042-43752dbf78a3)

## Running the Code
To run the code:
1. Clone this repository to your local machine.
2. Run the `hough.py` script.
3. Wait for the execution to complete. Note: The runtime may be around 10 minutes.
4. Optionally, modify parameters like discretization for faster execution.
5. Visualize the generated plots and results in the `results` folder.
