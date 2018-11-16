"""
Example for testing custom colors maps in OpenCV providing key color points
"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt

# This dictionary is only for debugging purposes:
dict_color = {0: "blue", 1: "green", 2: "red"}


def build_lut(cmap):
    """Builds look up table based on 'key colors' using np.linspace()"""

    lut = np.empty(shape=(256, 3), dtype=np.uint8)
    # Show for debugging purposes:
    print("----------")
    print(cmap)
    print("-----")

    max = 256
    # build lookup table:
    lastval, lastcol = cmap[0]
    for step, col in cmap[1:]:
        val = int(step * max)
        for i in range(3):
            print("{} : np.linspace('{}', '{}', '{}' - '{}' = '{}')".format(dict_color[i], lastcol[i], col[i], val,
                                                                            lastval, val - lastval))
            lut[lastval:val, i] = np.linspace(lastcol[i], col[i], val - lastval)

        lastcol = col
        lastval = val

    return lut


def apply_color_map_1(gray, cmap):
    """Applies a custom color map using cv2.LUT()"""

    lut = build_lut(cmap)
    s0, s1 = gray.shape
    out = np.empty(shape=(s0, s1, 3), dtype=np.uint8)

    for i in range(3):
        out[..., i] = cv2.LUT(gray, lut[:, i])
    return out


def apply_color_map_2(gray, cmap):
    """Applies a custom color map using cv2.applyColorMap()"""

    lut = build_lut(cmap)
    lut_reshape = np.reshape(lut, (256, 1, 3))
    im_color = cv2.applyColorMap(gray, lut_reshape)
    return im_color


def show_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Read grayscale image:
gray_img = cv2.imread('shades.png', cv2.IMREAD_GRAYSCALE)

# Create the dimensions of the figure and set title:
plt.figure(figsize=(14, 3))
plt.suptitle("Custom color maps based on key colors", fontsize=14, fontweight='bold')

# Show gray image:
show_with_matplotlib(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), "gray", 1)

# Apply the custom color map - (b,g,r) values:
custom_1 = apply_color_map_1(gray_img, ((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                                        (0.75, (255, 0, 60)), (1.0, (255, 0, 0))))

custom_2 = apply_color_map_1(gray_img, ((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                                        (0.75, (64, 128, 224)), (1.0, (0, 128, 255))))

custom_3 = apply_color_map_2(gray_img, ((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                                        (0.75, (255, 0, 60)), (1.0, (255, 0, 0))))

custom_4 = apply_color_map_2(gray_img, ((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                                        (0.75, (64, 128, 224)), (1.0, (0, 128, 255))))

# Display all the resulting images:
show_with_matplotlib(custom_1, "custom 1 using cv2.LUT()", 2)
show_with_matplotlib(custom_2, "custom 2 using cv2.LUT()", 3)
show_with_matplotlib(custom_3, "custom 3 using cv2.applyColorMap()", 5)
show_with_matplotlib(custom_4, "custom 4 using using cv2.applyColorMap()", 6)

# Show the Figure:
plt.show()
