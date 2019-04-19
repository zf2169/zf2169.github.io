---
layout: post
title: "Real-time "Stupid" Makeup - Face Recognition by Python - step by step"
date: 2019-04-19
---

# Installation
My Software Version:
- Python 3.6.3 :: Anaconda custom (64-bit)
- Windows 10

First, make sure you have the below packages already installed with Python bindings:
- CMake - use `pip install CMake` to install it
- dlib - Windows may have problems automatically installing the newest version, so you can specify the version and
use `pip install dlib==19.8.1`
<p align="center">
  <img width="600" src="https://zf2169.github.io/img/install.PNG">
</p>

Then, use `pip install face_recognition` to install face_recognition

# Usage
## Command-Line Interface
When you install face_recognition, you get two simple command-line programs:

face_recognition - Recognize faces in a photograph or folder full for photographs.

face_detection - Find faces in a photograph or folder full for photographs.

# Python Code

```
from PIL import Image, ImageDraw
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("two_people.jpg")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # Make the eyebrows into a nightmare
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # Gloss the lips
    d.polygon(face_landmarks['top_lip'], fill=(250, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(250, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    # Sparkle the eyes
    d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

    # Apply some eyeliner
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

pil_image.show()
```










