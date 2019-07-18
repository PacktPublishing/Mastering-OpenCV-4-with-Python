"""
Test script to play with OpenCV and Python
"""

import cv2

# use '$ python -m pydoc -w helloopencv' to generate documentation


def show_message():
    """this function returns a message

    """
    return "this function returns a message"


def load_image(path):
    """Loads the image given the path of the image to be loaded

    """
    return cv2.imread(path)


def show_image(image):
    """this function shows an image given the image and wait until a key is pressed

    """
    cv2.imshow("image", image)
    # waits forever for user to press any key
    cv2.waitKey(0)
    # close displayed windows
    cv2.destroyAllWindows()


def convert_to_grayscale(image):
    """this function converts a BGR image into a grayscale one

    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def write_image_to_disk(path, image):
    """this function writes to disk an image given the image to be written

    """
    cv2.imwrite(path, image)


if __name__ == "__main__":
    print("hellopencv.py is being run directly")

    # print the message returned by the function show_message
    print(show_message());

    # load image
    bgr_image = load_image("images/logo.png")

    # show image
    show_image(bgr_image)

    # convert image to black and white
    gray_image = convert_to_grayscale(bgr_image)

    # show image
    show_image(gray_image)

    # write gryscale image to disk
    write_image_to_disk("images/gray_logo.png", gray_image)
else:
    print("hellopencv.py is being imported into another module")
