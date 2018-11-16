"""
Example for testing custom colors maps in OpenCV and ploting the legend
"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_lut_image(cmap, height):
    """Builds the legend image"""

    lut = build_lut(cmap)
    image = np.repeat(lut[np.newaxis, ...], height, axis=0)

    return image


def build_lut(cmap):
    """Builds look up table based on 'key colors' using np.linspace()"""

    lut = np.empty(shape=(256, 3), dtype=np.uint8)
    max = 256
    # build lookup table:
    lastval, lastcol = cmap[0]
    for step, col in cmap[1:]:
        val = int(step * max)
        for i in range(3):
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
    lut2 = np.reshape(lut, (256, 1, 3))
    im_color = cv2.applyColorMap(gray, lut2)
    return im_color


def show_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Read grayscale image:
gray_img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# Create the dimensions of the figure and set title:
plt.figure(figsize=(14, 6))
plt.suptitle("Custom color maps based on key colors and legend", fontsize=14, fontweight='bold')

# Build the color maps (b,g,r) values:
custom_1 = apply_color_map_1(gray_img, ((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                                        (0.75, (255, 0, 60)), (1.0, (255, 0, 0))))

custom_2 = apply_color_map_2(gray_img, ((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                                        (0.75, (64, 128, 224)), (1.0, (0, 128, 255))))

# Build the legend images:
legend_1 = build_lut_image(((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                            (0.75, (255, 0, 60)), (1.0, (255, 0, 0))), 20)

legend_2 = build_lut_image(((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                            (0.75, (64, 128, 224)), (1.0, (0, 128, 255))), 20)

# Display all the resulting images:
show_with_matplotlib(legend_1, "", 1)
show_with_matplotlib(custom_1, "", 3)
show_with_matplotlib(legend_2, "", 2)
show_with_matplotlib(custom_2, "", 4)

# Show the Figure:
plt.show()
