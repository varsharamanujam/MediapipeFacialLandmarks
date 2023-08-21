# mediapie - Facial Landmarks and Iris Detection

`mediapie` is a project that focuses on detecting facial landmarks and eye points, including the iris. It consists of various Python scripts that utilize OpenCV, MediaPipe, and custom utilities.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files Overview](#files-overview)

## Installation

### Required Packages

Install the required packages using the following commands:

```bash
pip install cv2
pip install mediapipe
pip install numpy
```
## Usage
- `Facial Landmarks Detection:` Run facial_landmarks.py to detect facial landmarks and draw vertical and horizontal lines over the eye.
- `Iris Detection:` Run iris.py with the support of files in the custom directory to detect the iris of the eye.
- `Eye Direction Detection:` Run direction_detection.py to detect the direction of the eyes.

## Files Overview
- `facial_landmarks.py:` Detects facial landmarks and draws lines over the eye.
- `iris_landmark.py:` Supports iris detection.
- `iris.py:` Main script for iris detection.
- `direction_detection.py:` Detects the direction of the eyes.
- `meshpoints.py:` Contains mesh points related to eye and iris.
- `to find facial landmarks.py:` Finds facial landmarks.
- `utils.py:` Utility functions and color definitions.
- `videosource.py:` Class for handling video sources.



