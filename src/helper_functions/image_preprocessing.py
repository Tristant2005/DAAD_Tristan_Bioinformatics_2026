import numpy as np
import cv2
from skimage.util import img_as_float32
from skimage.color import rgb2gray
from skimage import exposure


def fill_stitching_corners(img, black_thresh=0.02, fill_value=None):
    """
    Detect black corner regions from image stitching and fill them with
    the image's actual background gray level, rather than leaving them
    as pure black (which distorts percentile-based contrast rescaling).
    """
    img = img.copy()
    height, width = img.shape

    # build a mask of "black" candidate pixels
    black_candidates = (img <= black_thresh).astype(np.uint8)

    # flood fill from each corner to find only the black regions that
    # are actually CONNECTED to a corner (not dark structures in the middle)
    corner_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)  # cv2 flood fill needs a 2px border
    flood_source = black_candidates.copy()

    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]

    combined_fill_mask = np.zeros((height, width), dtype=np.uint8)

    for y, x in corners:
        # this corner isn't actually black, skip it
        if flood_source[y, x] == 0:
            continue 

        # cv2.floodFill wants a nonzero-valued image to flood on
        mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        temp = flood_source.copy() * 255 

        cv2.floodFill(temp, mask, seedPoint=(x, y), newVal=128,
                    loDiff=0, upDiff=0, flags=8)

        # pixels that got set to 128 are the connected black region from this corner
        combined_fill_mask |= (temp == 128).astype(np.uint8)

    # estimate the background gray value from the NON-black pixels,
    # so the fill blends naturally instead of guessing a fixed constant
    if fill_value is None:
        non_black_pixels = img[combined_fill_mask == 0]
        fill_value = np.median(non_black_pixels) if len(non_black_pixels) > 0 else 0.5

    img[combined_fill_mask.astype(bool)] = fill_value

    return img, combined_fill_mask.astype(bool)


def preprocess_pipeline(img, gaussian_sigma=0.5, percentiles=(2, 98), is_stitched=False):
    img = img_as_float32(img)

    # Step 1: convert to grayscale if needed
    if len(img.shape) == 3:
        gray = rgb2gray(img)
    else:
        gray = img

    # Step 1.5: fill stitched-image black corners BEFORE contrast rescaling,
    # since percentile stretching is what actually breaks on these images --
    # a large black corner skews the p2/p98 computation for the whole image
    if is_stitched:
        gray, _ = fill_stitching_corners(gray)

    # Step 2: Gaussian denoise
    # denoised = gaussian(gray, sigma=gaussian_sigma)

    # Step 3: percentile contrast rescaling
    p_low, p_high = percentiles
    p2, p98 = np.percentile(gray, (p_low, p_high))
    processed = exposure.rescale_intensity(gray, in_range=(p2, p98))

    return processed