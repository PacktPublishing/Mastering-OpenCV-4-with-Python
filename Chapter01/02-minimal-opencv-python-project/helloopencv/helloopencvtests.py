"""
Tests for helloopencv
"""

# To run the tests: 'py.test -s -v helloopencvtests.py'

from helloopencv import show_message
from helloopencv import load_image
from helloopencv import write_image_to_disk
from helloopencv import convert_to_grayscale

import cv2


def test_show_message():
    """Test for show_message

    """
    print("testing show_message")
    assert show_message() == "this function returns a message"


def test_load_image():
    """Test for load_image

    """
    print("testing load_image")
    bgr_image = load_image("images/logo.png")
    assert bgr_image is not None


def test_write_image_to_disk():
    """Test for write_image_to_disk

    """
    print("testing write_image_to_disk")
    # load the image from disk
    bgr_image = load_image("images/logo.png")
    # write image to disk
    write_image_to_disk("images/temp.png", bgr_image)
    # load the image temp from disk
    temp = load_image("images/temp.png")
    # now we check that the two images are equal
    assert bgr_image.shape == temp.shape
    difference = cv2.subtract(bgr_image, temp)
    b, g, r = cv2.split(difference)
    assert cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0


def test_convert_to_grayscale():
    """Test for write_image_to_disk

    """
    print("testing test_convert_to_grayscale")
    # load the image from disk
    bgr_image = load_image("images/logo.png")
    # convert image to black and white
    gray_image = convert_to_grayscale(bgr_image)
    bgr_height, bgr_width, bgr_channels = bgr_image.shape
    gray_height, gray_width = gray_image.shape
    assert bgr_height == gray_height and bgr_width == gray_width







