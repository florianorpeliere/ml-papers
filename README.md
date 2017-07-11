# Machine Learning Papers

## Introduction

* [Machine Learning is Fun!](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471)

## Computer Vision

### Dataset

* [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf): New image recognition, segmentation and capturing dataset. [Link to COCO Dataset](http://mscoco.org/home/)
  * Image annonation with Amazon's Mechanical Turk
  * Bounding Box Detection with DPMv5-P/DPMv5-C
* [YouTube-BoundingBoxes: A Large High-Precision
Human-Annotated Data Set for Object Detection in Video](https://arxiv.org/pdf/1702.00824.pdf): New large-scale data set of video URLs with densely-sampled object bounding box annotations. (Approximately 380,000 video segments about 19s
long)
* [AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions](https://arxiv.org/pdf/1705.08421.pdf): New video dataset. Every person is localized using a bounding box and the attached labels correspond to actions being performed by the person. There is one action corresponding to the **pose of the person** (whether he or she is standing, sitting, walking, swimming etc.) and there may be additional actions corresponding to **interactions with objects** or **human-human interactions**. The main differences with existing video datasets are: 
  * the definition of atomic visual actions, which avoids collecting data for each and every complex action
  * precise spatio-temporal annotations with possibly multiple annotations for each human
  * the use of diverse, realistic video material (movies)



### Meta-architecture & feature extractor
  
* [Google research blog: Supercharge your Computer Vision models with the TensorFlow Object Detection API](https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html): Google won COCO Detection challenge with they in-house object detection. This system is available via [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)
* [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf): Object detection with a single deep neural network [Github Code](https://github.com/weiliu89/caffe/tree/ssd)
* [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/pdf/1611.10012.pdf): Guide for selecting a detection architecture that achieves the right speed/memory/accuracy balance for a given application and platform. They compare SSD, [Faster R-CNN](https://arxiv.org/abs/1506.01497) and [R-FCN](https://arxiv.org/pdf/1605.06409.pdf) meta-achitecture with some architecural configuation like feature extractor, matching and Box encoding.
<br>List of feature extractor
  * [Inception Resnet V2](https://arxiv.org/pdf/1602.07261.pdf): Deep convolutional networks. [Blog article](https://research.googleblog.com/2016/08/improving-inception-and-image.html)
  * [Inception architecture v3](https://arxiv.org/pdf/1512.00567.pdf): Deep convolutional networks. [TensorFlow Github code](https://github.com/tensorflow/models/blob/master/slim/nets/inception_v3.py)
  * [VGG-16](https://arxiv.org/pdf/1409.1556.pdf)
  * [MobileNet](https://arxiv.org/pdf/1704.04861.pdf)
  * [Inception V2](https://arxiv.org/pdf/1502.03167.pdf)
  * [ResNet-101](https://arxiv.org/pdf/1512.03385.pdf)
* [Spatially Adaptive Computation Time for Residual Networks](https://arxiv.org/pdf/1612.02297.pdf): This paper proposes a deep learning architecture based on Residual Network that dynamically adjusts the number of executed layers for the regions of the image. 

* [Tensorflow Object Detection API example](https://cloud.google.com/blog/big-data/2017/06/training-an-object-detector-using-cloud-machine-learning-engine)

### Models
* Python ([dlib](http://dlib.net/))
  * [dlib-models](https://github.com/davisking/dlib-models)
  
### Others

* [Beyond Skip Connections: Top-Down Modulation for Object Detection](https://arxiv.org/pdf/1612.06851.pdf)
* [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)
* [Blog: Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/)

## Face recognition

### Dataset

* [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/): Faces dataset

### Feature extractor

* [FaceNet: A Unified Embedding for Face Recognition and Clustering](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf): Directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. The benefit of this approach is much greater representational efficiency: they achieve state-of-the-art face recognition performance using only 128-bytes per face.

### Models

* [Python: OpenFace](http://cmusatyalab.github.io/openface/)
