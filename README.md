# Image Alignment using ORB (Feature Detection & Homography)

## Project overview

This repository contains code and a Jupyter notebook to detect keypoints and descriptors in images, match them between two images, compute a transformation matrix (homography/affine), and align one image to another. The implementation uses OpenCV's ORB feature detector and descriptor and demonstrates a full pipeline from feature extraction to image warping.

### Tasks

* **Task 1:** Determine keypoint locations and descriptors for an image (and the image to be aligned).
* **Task 2:** Determine matching keypoints between the two images.
* **Task 3:** Compute the transformation matrix using matched keypoints (e.g., homography via RANSAC or affine transform).
* **Task 4:** Use the transformation matrix to align (warp) the image to be aligned to the reference image.

## Files

* `CV_ORB_to_align_images.ipynb` — Jupyter notebook with the full step-by-step implementation, visualizations, and explanations.
* (Optional) `align_images.py` — A lightweight script (if included) to run the pipeline from the command line.
* `examples/` — Example input images and results (aligned outputs and match visualizations).

## Requirements

* Python 3.8+ recommended
* Libraries:

  * `opencv-python` (cv2)
  * `numpy`
  * `matplotlib` (for visualization)
  * `scikit-image` or `scipy` (optional — for additional transform utilities)

Install dependencies with pip:

```bash
pip install opencv-python numpy matplotlib
# optional
pip install scikit-image scipy
```

## Notebook usage

1. Open the notebook in Jupyter or VS Code:

```bash
jupyter notebook CV_ORB_to_align_images.ipynb
```

2. Steps inside the notebook:

   * Load the reference image and the image-to-align.
   * Convert to grayscale and (optionally) apply preprocessing (resize, blur, histogram equalization).
   * Detect keypoints and compute descriptors using ORB.
   * Match descriptors (e.g., BFMatcher with Hamming distance for ORB).
   * Filter matches (ratio test / cross-check / symmetric match) and select good matches.
   * Estimate transformation matrix: `cv2.findHomography()` with RANSAC (for perspective) or `cv2.estimateAffinePartial2D()` for affine.
   * Warp the image using `cv2.warpPerspective()` or `cv2.warpAffine()` to align to the reference image.
   * Visualize keypoints, matches, inliers, and the final aligned image.

## Example (script-style)

If you prefer a script (pseudo-usage):

```bash
python align_images.py --ref ref.jpg --src to_align.jpg --out aligned.jpg --visualize matches.png
```

Inside `align_images.py` the pipeline follows these high-level steps:

```python
# 1. load images
# 2. detect ORB keypoints + descriptors
# 3. match using cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
# 4. apply Lowe's ratio test to keep good matches
# 5. compute homography: cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# 6. warp: cv2.warpPerspective(src_img, homography, (w_ref, h_ref))
```

## Tips & troubleshooting

* ORB is fast and free to use (non-patented), but may fail for images with very low texture or heavy illumination differences. Try:

  * Increasing number of features in ORB (`cv2.ORB_create(nfeatures=2000)`).
  * Preprocessing (histogram equalization, CLAHE, or contrast adjustments).
  * Using different match filtering strategies (cross-check or ratio test thresholds).
  * Switching to SIFT / SURF (if license allows) or using deep-learning based feature matchers for difficult cases.
* If homography estimation fails (not enough inliers), check that the images actually share overlapping content and that correspondences are correct.

## Expected outputs

* Visualizations of detected keypoints on both images.
* Match lines between images showing matched keypoints.
* Image showing only inlier matches (after RANSAC).
* The aligned (warped) image aligned to the reference view.

## References

* OpenCV documentation: Feature detection and description, `cv2.ORB`, `cv2.BFMatcher`, `cv2.findHomography`, `cv2.warpPerspective`.
* Tutorials on feature matching and homography estimation.

