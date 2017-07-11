# Machine Learning Papers

## Computer Vision

* [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf): New image recognition, segmentation and capturing dataset. [Link to COCO Dataset](http://mscoco.org/home/)
  * Image annonation with Amazon's Mechanical Turk
  * Bounding Box Detection with DPMv5-P/DPMv5-C
* [Google research blog: Supercharge your Computer Vision models with the TensorFlow Object Detection API](https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html): Google won COCO Detection challenge with they in-house object detection. This system is available via [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)
* [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf): Object detection with a single deep neural network [Github Code](https://github.com/weiliu89/caffe/tree/ssd)
* [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/pdf/1611.10012.pdf): Guide for selecting a detection architecture that achieves the right speed/memory/accuracy balance for a given application and platform. They compare SSD, [Faster R-CNN](https://arxiv.org/abs/1506.01497) and R-FCN meta-achitecture with some architecural configuation like feature extractor, matching and Box encoding.
List of feature extractor
  * [Inception Resnet V2](https://arxiv.org/pdf/1602.07261.pdf): Deep convolutional networks. [Blog article](https://research.googleblog.com/2016/08/improving-inception-and-image.html)
  * [Inception architecture v3](https://arxiv.org/pdf/1512.00567.pdf): Deep convolutional networks. [TensorFlow Github code](https://github.com/tensorflow/models/blob/master/slim/nets/inception_v3.py)
  * VGG-16
  * [MobileNet](https://arxiv.org/pdf/1704.04861.pdf)
  * Inception V2
  * [ResNet-101](https://arxiv.org/pdf/1512.03385.pdf)

* [Tensorflow Object Detection API example](https://cloud.google.com/blog/big-data/2017/06/training-an-object-detector-using-cloud-machine-learning-engine)
