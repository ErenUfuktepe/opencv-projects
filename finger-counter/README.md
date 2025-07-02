# Finger Counter with OpenCV

This project detects and counts the number of fingers held up in front of the webcam using OpenCV and basic contour analysis techniques. It's a fun way to explore hand detection, contour processing, and gesture recognition in real time.

## Why This Project?

I built this project as part of my learning journey with OpenCV. It helped me explore how to:

- Detect hand regions using contours and convex hulls
- Identify fingers based on defects in the hand's contour
- Perform real-time gesture analysis with minimal dependencies

---

## What It Can Do

- Detects a hand in the webcam feed using skin color segmentation
- Finds the largest contour and calculates convex hull and convexity defects
- Uses geometry to estimate the number of raised fingers 
- Displays the finger count live on screen

---

## Requirements

- Python 3.7+
- OpenCV
- NumPy
