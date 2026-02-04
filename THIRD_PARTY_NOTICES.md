# Third-Party Notices

This project bundles third-party files. They are NOT authored by the project maintainer.

## OpenCV sample model definition: `deploy.prototxt`

- Source: OpenCV repository, `samples/dnn/face_detector/deploy.prototxt`
- Upstream project: OpenCV (Open Source Computer Vision Library)
- License: OpenCV 4.5.0 and higher are licensed under Apache License 2.0.
- References:
  - OpenCV licensing overview: https://opencv.org/license/
  - OpenCV LICENSE (Apache-2.0): https://github.com/opencv/opencv/blob/4.x/LICENSE

## OpenCV DNN face detector weights: `res10_300x300_ssd_iter_140000.caffemodel`

- Common OpenCV reference distribution:
  - OpenCV sample models list points to an `opencv/opencv_3rdparty` raw URL (branch/tag `dnn_samples_face_detector_20170830`).
- References:
  - OpenCV sample `models.yml` (URL reference): https://github.com/opencv/opencv/blob/4.x/samples/dnn/models.yml
  - Example notebook using the same OpenCV URLs: https://colab.research.google.com/github/dortmans/ml_notebooks/blob/master/face_detection.ipynb

### License status for the weights file
The redistribution license for this specific `.caffemodel` file may not be clearly stated in the public OpenCV 3rdparty references.
OpenCV community guidance indicates you may need to confirm licensing/provenance with the contributor or upstream source before redistributing.
Reference discussion: https://forum.opencv.org/t/license-of-3rd-party-dnn-model/4300


