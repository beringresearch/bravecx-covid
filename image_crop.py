import tensorflow as tf
import numpy as np

@tf.function
def crop_image_to_mask(img, mask,
    crop_shape=(299, 299),
    crop_masked_img=True, soft_mask=True):
    """Reads an image and a mask from disk using the provided read functions
    and returns a randomly selected image crop of the requested size that is centered around a
    segmented pixel.

    The mask is assumed to be scaled within the usual range for the provided dtype (e.g.
    0-255 for tf.uint8).

    Args:
        crop_shape: tuple(2). Requested crop size.
        crop_masked_img: boolean. If true, crops masked image rather than original.
        soft_mask: boolean. If true, the soft image mask is applied to the image,
            rather than a hard one.
    """
    mask = tf.convert_to_tensor(mask)
    mask /= 255
    mask_hard = tf.cast(tf.math.round(mask), dtype=img.dtype)

    if len(tf.shape(mask_hard)) < 3:
        mask_hard = tf.expand_dims(mask_hard, axis=-1)

    if crop_masked_img:
        if soft_mask:
            img_to_crop = tf.cast(tf.cast(img, tf.float32) * mask, img.dtype)
        else:
            img_to_crop = img * mask_hard
    else:
        img_to_crop = img

    crop_center_coords = select_random_segmented_pixel(mask_hard)
    cropped_img = centered_img_crop(img_to_crop, crop_center_coords, crop_shape)
    return cropped_img

@tf.function
def centered_img_crop(img, center_pixel_coords, crop_shape):
    """Extracts a crop from an image centered around the coordinates provided
    in center_pixel_coords.

    If the requested crop would extend beyond the edge of the image it will be constrained
    to fit inside.

    Args:
        img - image Tensor
        center_pixel_coords: tuple/list (height, width) with coordinates to center of crop
        crop_shape: tuple/list of (height, width) with dimensions of the crop to take

    Returns:
        img_crop: Tensor
    """

    center_pixel_coords = tf.cast(center_pixel_coords, tf.int32)
    crop_shape = tf.cast(crop_shape, tf.int32)

    min_val = tf.constant(0, dtype=tf.int32)
    two = tf.constant(2, dtype=tf.int32)

    offset_height = tf.math.maximum(center_pixel_coords[0] - crop_shape[0] // two, min_val)
    offset_height = tf.math.minimum(offset_height, tf.shape(img)[0] - crop_shape[0])
    offset_width = tf.math.maximum(center_pixel_coords[1] - crop_shape[1] // two, min_val)
    offset_width = tf.math.minimum(offset_width, tf.shape(img)[1] - crop_shape[1])

    img_crop = tf.image.crop_to_bounding_box(
        img, offset_height=offset_height, offset_width=offset_width,
        target_height=crop_shape[0], target_width=crop_shape[1])

    return img_crop

@tf.function
def select_random_segmented_pixel(mask):
    """Takes a binary mask and outputs the coordinates of a random positive point.

    Args:
        mask - 2D Tensor or 3D where one dimension is size 1.
        Assumes binary values [0, 1] where 1 denotes the candidates for sampling.
    """
    mask = tf.squeeze(mask)

    positive_segmentation_tensor = tf.ones(tf.shape(mask), dtype=mask.dtype)
    segmented_pixel_coords = tf.where(tf.equal(mask, positive_segmentation_tensor))

    random_coord_idx = tf.random.uniform([], maxval=tf.shape(segmented_pixel_coords)[0], dtype=tf.int32)
    random_segmented_coord = segmented_pixel_coords[random_coord_idx]
    return random_segmented_coord