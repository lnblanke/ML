import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

def preprocess(dataset: tf.data.Dataset, batch_size: int, n_grids = 7) -> tf.data.Dataset:
    # Resize all images so that they can be batched
    resize = tf.keras.layers.Resizing(224, 224)
    images, labels = [], []
    
    for x in dataset:
        (img, bbox, label) = x["image"], x["objects"]["bbox"], x["objects"]["label"]
        img_h, img_w = img.shape[-3: -1]

        # Each grid contains 25 elements: (center_x, center_y, width, height, whether contains label 0 ~ 19, whether contains object)
        true_label = np.zeros((n_grids, n_grids, 25), dtype = np.float32)

        for ([y1, x1, y2, x2], l) in zip(bbox, label):
            # The width and the height of the box w.r.t. to the image
            box_w, box_h = x2 - x1, y2 - y1

            # The center position of the rectangle w.r.t. to the image
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # The grid that the center is in
            grid_x, grid_y = tf.cast(center_x * n_grids, tf.int32), tf.cast(center_y * n_grids, tf.int32)

            # Modify the center of the rectangle from w.r.t. the whole image to w.r.t. to the grid
            center_x = (center_x - tf.cast(grid_x / n_grids, tf.float32)) * n_grids
            center_y = (center_y - tf.cast(grid_y / n_grids, tf.float32)) * n_grids

            true_label[grid_x, grid_y, :4] = [center_x, center_y, box_w, box_h]
            true_label[grid_x, grid_y, l + 4] = 1
            true_label[grid_x, grid_y, -1] = 1

        images.append(resize(img) / 255.0 - .5)
        labels.append(true_label)

    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size, drop_remainder = True)

'''
    Get VOC 2007 dataset and preprocess the data to better fit the model.
'''
def get_data(batch_size, n_grids = 7):
    train_ds, val_ds, test_ds = tfds.load("voc", split = ["train", "validation", "test"], shuffle_files = True)

    return (preprocess(train_ds.shuffle(batch_size), batch_size, n_grids), 
            preprocess(val_ds, batch_size, n_grids), 
            preprocess(test_ds, batch_size, n_grids))