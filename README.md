# Mastering OpenCV 4 with Python

<a href="https://www.packtpub.com/application-development/mastering-opencv-4-python?utm_source=github&utm_medium=repository&utm_campaign=9781789344912 "><img src="https://prod.packtpub.com/media/catalog/product/cache/a22c7d190d97ca25f5f1089471ab8502/9/7/978178934_cover.png" alt="Mastering OpenCV 4 with Python" height="256px" align="right"></a>

This is the code repository for [Mastering OpenCV 4 with Python](https://www.packtpub.com/application-development/mastering-opencv-4-python?utm_source=github&utm_medium=repository&utm_campaign=9781789344912 ), published by Packt.

**A practical guide covering topics from image processing, augmented reality to deep learning with OpenCV 4 and Python 3.7**

## What is this book about?
OpenCV is considered to be one of the best Open Source Computer Vision and machine learning software libraries. It helps developers build complete projects on image processing, motion detection, and image segmentation. OpenCV for Python enables you to run computer vision algorithms smoothly in real time, combining the best of the OpenCV C++ API and the Python language.

This book covers the following exciting features:
* Handle files and images, and explore various image processing techniques 
* Explore image transformations like translation, resizing, and cropping 
* Gain insights into building histograms 
* Brush up on contour detection, filtering, and drawing 
* Work with augmented reality and 3D visualization frameworks 
* Work with machine learning, deep learning, and neural network algorithms 

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1789344913) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
channels = cv2.split(img)
eq_channels = []
for ch in channels:
eq_channels.append(cv2.equalizeHist(ch))
```

## Code Testing Specifications
*Mastering OpenCV 4 with Python* requires some installed packages, which you can see next.

* Chapter01 (*Setting Up OpenCV*): ``opencv-contrib-python``
* Chapter02 (*Image Basics in OpenCV*): ``opencv-contrib-python matplotlib``
* Chapter03 (*Handling Files and Images*): ``opencv-contrib-python matplotlib``
* Chapter04 (*Constructing Basic Shapes in OpenCV*): ``opencv-contrib-python matplotlib``
* Chapter05 (*Image Processing Techniques*): ``opencv-contrib-python matplotlib``
* Chapter06 (*Constructing and Building Histograms*): ``opencv-contrib-python matplotlib``
* Chapter07 (*Thresholding Techniques*): ``opencv-contrib-python matplotlib scikit-image, scipy``
* Chapter08 (*Contours Detection, Filtering, and Drawing*): ``opencv-contrib-python matplotlib``
* Chapter09 (*Augmented Reality*): ``opencv-contrib-python matplotlib``
* Chapter10 (*Machine Learning with OpenCV*): ``opencv-contrib-python matplotlib``
* Chapter11 (*Face Detection, Tracking, and Recognition*): ``opencv-contrib-python matplotlib dlib face-recognition cvlib requests progressbar keras tensorflow``
* Chapter12 (*Introduction to Deep Learning*): ``opencv-contrib-python matplotlib tensorflow keras``
* Chapter13 (*Mobile and Web Computer Vision with Python and OpenCV*): ``opencv-contrib-python matplotlib flask tensorflow keras requests pillow``

Make sure that the version numbers of your installed packages are equal to, or greater than, versions specified below to ensure the code examples run correctly. If you want to install the exact versions this book was tested on, include the version when installing from pip.

* Install opencv-contrib-python:

```
pip install opencv-contrib-python==4.0.0.21
```
It should be noted that OpenCV requires: ``numpy`` 

``numpy-1.16.1`` has been installed when installing ``opencv-contrib-python==4.0.0.21`` 

 * Install matplotlib:
 
```
pip install matplotlib==3.0.2
```
It should be noted that matplotlib requires: ``kiwisolver pyparsing six cycler python-dateutil``

``cycler-0.10.0 kiwisolver-1.0.1  pyparsing-2.3.1 python-dateutil-2.8.0 six-1.12.0`` have been installed when installing ``matplotlib==3.0.2``

 * Install scikit-image:
```
pip install scikit-image==0.14.2
```
It should be noted that scikit-image requires: ``cloudpickle decorator networkx numpy toolz dask pillow PyWavelets six``

``PyWavelets-1.0.1 cloudpickle-0.8.0 dask-1.1.1 decorator-4.3.2 networkx-2.2 numpy-1.16.1 pillow-5.4.1 six-1.12.0 toolz-0.9.0`` have been installed when installing ``scikit-image==0.14.2``

 * Install scipy:
```
pip install scipy==1.2.1 
```
It should be noted that scipy requires: ``numpy``

``numpy-1.16.1`` has been installed when installing ``scipy==1.2.1``

 * Install dlib:
```
pip install dlib==19.8.1 
```

 * Install face-recognition:
```
pip install face-recognition==1.2.3
```
It should be noted that face-recognition requires: ``dlib Click numpy face-recognition-models pillow``

``dlib-19.8.1 Click-7.0 face-recognition-models-0.3.0 pillow-5.4.1`` have been installed when installing ``face-recognition==1.2.3``

 * Install cvlib:
```
pip install cvlib==0.1.8
```

 * Install requests:
```
pip install requests==2.21.0
```

It should be noted that requests requires: ``urllib3 chardet certifi idna``

``urllib3-1.24.1 chardet-3.0.4 certifi-2018.11.29 idna-2.8`` have been installed when installing ``requests==2.21.0``

 * Install progressbar:
```
pip install progressbar==2.5 
```

 * Install keras:
 
```
pip install keras==2.2.4
``` 
It should be noted that keras requires: ``numpy six h5py keras-applications scipy keras-preprocessing pyyaml``

``h5py-2.9.0 keras-applications-1.0.7 keras-preprocessing-1.0.9 numpy-1.16.1 pyyaml-3.13 scipy-1.2.1 six-1.12.0`` have been installed when installing ``keras==2.2.4``

 * Install tensorflow:
 
```
pip install tensorflow==1.12.0 
```
It should be noted that tensorflow requires: ``termcolor numpy wheel gast six setuptools protobuf markdown grpcio werkzeug tensorboard absl-py h5py keras-applications keras-preprocessing, astor``

``termcolor-1.1.0 numpy-1.16.1 wheel-0.33.1 gast-0.2.2 six-1.12.0 setuptools-40.8.0 protobuf-3.6.1 markdown-3.0.1 grpcio-1.18.0 werkzeug-0.14.1 tensorboard-1.12.2 absl-py-0.7.0 h5py-2.9.0 keras-applications-1.0.7 keras-preprocessing-1.0.9 astor-0.7.1`` have been installed when installing ``tensorflow==1.12.0``

 * Install flask:
```
pip install flask==1.0.2
```

It should be noted that flask requires: ``Werkzeug click itsdangerous MarkupSafe Jinja2``

``Jinja2-2.10 MarkupSafe-1.1.1 Werkzeug-0.14.1 click-7.0 itsdangerous-1.1.0`` have been installed when installing ``flask==1.0.2``

**Following is what you need for this book:**
This book is designed for computer vision developers, engineers, and researchers who want to develop modern computer vision applications. Basic experience of OpenCV and Python programming is a must.

With the following software and hardware list you can run all code files present in the book (Chapter 1-13).
### Software and Hardware List
| Chapter | Software required | Hardware required                        |
| --------| ----------------- | ------------------------------------------------------------------------------------------------------- |
| 1-13    | Python            | Either 32-bit or 64-bit architecture, 2+ GHz CPU, 4 GB RAM, At least 10 GB of hard disk space available |
| 1-13    | OpenCV            | Either 32-bit or 64-bit architecture, 2+ GHz CPU, 4 GB RAM, At least 10 GB of hard disk space available |
| 1-13    | PyCharm           | Either 32-bit or 64-bit architecture, 2+ GHz CPU, 4 GB RAM, At least 10 GB of hard disk space available |

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://www.packtpub.com/sites/default/files/downloads/9781789344912_ColorImages.pdf).

### Related products
* Learn OpenCV 4 By Building Projects - Second Edition [[Packt]](https://www.packtpub.com/application-development/learn-opencv-4-building-projects-second-edition?utm_source=github&utm_medium=repository&utm_campaign=9781789341225 ) [[Amazon]](https://www.amazon.com/dp/1789341221)

* Mastering OpenCV 4 - Third Edition [[Packt]](https://www.packtpub.com/application-development/mastering-opencv-4-third-edition?utm_source=github&utm_medium=repository&utm_campaign=9781789533576 ) [[Amazon]](https://www.amazon.com/dp/1789533570)

* OpenCV Computer Vision with Python [[Packt]](https://www.packtpub.com/application-development/opencv-computer-vision-python)  [[Amazon]](https://www.amazon.com/dp/1782163921)

* OpenCV: Computer Vision Projects with Python [[Packt]](https://www.packtpub.com/application-development/opencv-computer-vision-projects-python) [[Amazon]](https://www.amazon.com/dp/1787125491)

* Augmented Reality for Developers [[Packt]](https://www.packtpub.com/web-development/augmented-reality-developers) [[Amazon]](https://www.amazon.com/dp/1787286436)

* Deep Learning with Python and OpenCV [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-python-and-opencv)  [[Amazon]](https://www.amazon.com/dp/1788627326)

* Deep Learning with Keras [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-keras)  [[Amazon]](https://www.amazon.com/dp/1787128423)

* Getting Started with TensorFlow  [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/getting-started-tensorflow)  [[Amazon]](https://www.amazon.com/dp/1786468573)

* Mastering Flask Web Development - Second Edition [[Packt]](https://www.packtpub.com/web-development/mastering-flask-web-development-second-edition)  [[Amazon]](https://www.amazon.com/dp/1788995406)


## Get to Know the Author
**Alberto Fernández Villán**
is a software engineer with more than 12 years of experience in developing innovative solutions. In the last couple of years, he has been working in various projects related to monitoring systems for industrial plants, applying both Internet of Things (IoT) and big data technologies. He has a Ph.D. in computer vision (2017), a deep learning certification (2018), and several publications in connection with computer vision and machine learning in journals such as Machine Vision and Applications, IEEE Transactions on Industrial Informatics, Sensors, IEEE Transactions on Industry Applications, IEEE Latin America Transactions, and more. As of 2013, he is a registered and active user (albertofernandez) on the Q&A OpenCV forum.


### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.


