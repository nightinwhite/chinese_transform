import functools

import tensorflow as tf
from My_Log import Log_Manager

import inception_preprocessing


def base_norm(single_img):
    Log_Manager.print_info("Base_norm")
    with tf.variable_scope("Base_norm"):
        single_img = single_img / 255.
        single_img = single_img - 0.5
        return single_img

def slice_and_resize(single_img):
    with tf.variable_scope('slice_and_resize'):
        height = single_img.get_shape().dims[0].value
        width = single_img.get_shape().dims[1].value

        # Random crop cut from the street sign image, resized to the same size.
        # Assures that the crop is covers at least 0.8 area of the input image.
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(single_img),
            bounding_boxes=tf.zeros([0, 0, 4]),
            min_object_covered=0.8,
            aspect_ratio_range=[0.8, 1.2],
            area_range=[0.8, 1.0],
            use_image_if_no_bounding_boxes=True)
        distorted_image = tf.slice(single_img, bbox_begin, bbox_size)

        # Randomly chooses one of the 4 interpolation methods
        distorted_image = inception_preprocessing.apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize_images(x, [height, width], method),
            num_cases=4)
        distorted_image.set_shape([height, width, 3])
        return distorted_image

def distorting_color(single_image):
    with tf.variable_scope('distorting_color'):
        distorted_image = inception_preprocessing.apply_with_random_selector(
            single_image,
            functools.partial(
                inception_preprocessing.distort_color, fast_mode=False),
            num_cases=4)
        distorted_image = tf.clip_by_value(distorted_image, -1.5, 1.5)
        return distorted_image



