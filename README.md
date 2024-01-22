# SCARED Structure

- **keyframe**
  - `endoscope_calibration.yaml`: contains intrinsic matrix and R, T between two cameras
  - `left(right)_depth_map.tiff`: contains 3D depth info of the left image, captured using structured light
  - `Left(Right)_Image.png`: single rgb image
  - `point_cloud.obj`
  - `data`:
    - `scene_points`: contains .tiff files of depth info (left stacked on right), warped from the first frame
    - `frame_data`: contains .json files of camera info, including intrinsic matrix and R, T, as well as camera poses
    - `rgb.mp4`: video file
    - `left(right)_finalpass`: contains rectified rgb images in png
    - `reprojection_data`: contains .json files of reprojection matrix Q
    - `disparity_left(right)`: contains .tiff files of disparity map
    - `occl_left(right)`: contains .png files of occlusion map
