import numpy as np
from skimage.color import gray2rgb, hsv2rgb, rgb2hsv


def colorize_image_with_labels(image, labels, saturation=.4, ret_hues=True):
    """Convert a 2D grayscale image to a (label-defined) colored RGB image

    Parameters
    ----------
    image: 2d ndarray unint8
        grayscale image
    labels: 2d ndarray integer
        mask image for colorization
    saturation: float
        color saturation (in HSV)
    ret_hues: bool
        return color hue dictionary
    """
    hsv_img = rgb2hsv(np.array(gray2rgb(image), dtype=np.uint8))
    hue_img = hsv_img[:, :, 0]
    sat_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]
    # add the color
    num_labels = np.max(labels)
    hues = np.linspace(0, 1, num_labels, endpoint=False)
    hue_dict = {}
    for ii in range(num_labels):
        mask = labels == ii + 1
        if mask.sum():
            hue_img[mask] = hues[ii]
            sat_img[mask] = saturation
            hue_dict[ii + 1] = hues[ii]
    hsv_img_2 = np.array(np.dstack((hue_img, sat_img, value_img)))
    rgb = hsv2rgb(hsv_img_2)
    rgb = np.array(rgb * 255, dtype=np.uint8)
    if ret_hues:
        return rgb, hue_dict
    else:
        return rgb


def colorize_image_with_mask(image, mask, mask2=None, saturation=.4):
    """Convert a 2D grayscale image to a (masked-defined) colored RGB image

    Parameters
    ----------
    image: 2d ndarray unint8
        grayscale image
    mask: 2d ndarray boolean
        mask image for colorization
    mask2: 2d ndarray boolean
        optional second mask image for comparison with first mask image
    saturation: float
        color saturation (in HSV)
    """
    hsv_img = rgb2hsv(np.array(gray2rgb(image), dtype=np.uint8))
    hue_img = hsv_img[:, :, 0]
    sat_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]
    hue_img[mask] = .5
    hue_img[~mask] = .2
    if mask2 is not None:
        mask_diff = np.array(mask, dtype=int) - mask2
        hue_img[mask_diff < 0] = 0
        hue_img[mask_diff > 0] = .8
    sat_img[:] = saturation
    hsv_img_2 = np.array(np.dstack((hue_img, sat_img, value_img)))
    rgb = hsv2rgb(hsv_img_2)
    rgb = np.array(rgb * 256, dtype=np.uint8)
    return rgb
