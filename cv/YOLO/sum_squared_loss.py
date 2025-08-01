import tensorflow as tf

class SumSquaredLoss(tf.keras.losses.Loss):
    def __init__(self, coord = 5, noobj = .5, n_grids = 7):
        super().__init__()

        self.coord = coord
        self.noobj = noobj
        self.n_grids = n_grids
      
    def __call__(self, y_true, y_pred, sample_weight = None):
        # Reshape the prediction to (batch_size, grid_width, grid_height, # of prediction for each grid)
        y_pred = tf.cast(tf.reshape(y_pred, (-1, self.n_grids, self.n_grids, 30)), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # Separate the prediction into prediction for position and label
        pred_bbox = tf.reshape(y_pred[:, :, :, :10], (-1, self.n_grids, self.n_grids, 2, 5))
        pred_class = y_pred[:, :, :, 10:]

        # Get true labels
        (true_bbox, true_label, mask) = y_true[:, :, :, :4], y_true[:, :, :, 4: -1], y_true[:, :, :, -1]
        
        [cx_pred, cy_pred, w_pred, h_pred, complete_c_pred] = tf.unstack(pred_bbox, axis = -1)
        [cx_true, cy_true, w_true, h_true] = tf.unstack(true_bbox, axis = -1)
        
        # Get the coordinates of upper left and lower right corners for each intersect
        x_1 = tf.maximum(cx_pred - w_pred / 2, tf.expand_dims(cx_true - w_true / 2, axis = -1))
        y_1 = tf.maximum(cy_pred - h_pred / 2, tf.expand_dims(cy_true - h_true / 2, axis = -1))
        x_2 = tf.minimum(cx_pred + w_pred / 2, tf.expand_dims(cx_true + w_true / 2, axis = -1))
        y_2 = tf.minimum(cy_pred + h_pred / 2, tf.expand_dims(cy_true + h_true / 2, axis = -1))

        # Calculate IOU score
        intersect = tf.maximum(x_2 - x_1, tf.zeros_like(x_1)) / self.n_grids * tf.maximum(y_2 - y_1, tf.zeros_like(y_1)) / self.n_grids
        union = h_pred * w_pred + tf.expand_dims(h_true * w_true, axis = -1) - intersect
        iou = tf.math.divide_no_nan(intersect, union)
        print(x_1[0, 2, 5], y_1[0, 2, 5], x_2[0, 2, 5], y_2[0, 2, 5])

        # Get the bbox with highest IOU score for each grid
        arg = tf.argmax(iou, axis = -1, output_type = tf.int32)

        # Retrieve the corresponding bboxes
        pred_bbox = tf.where(tf.cast(arg[:, :, :, None], tf.bool), pred_bbox[:, :, :, 1, :], pred_bbox[:, :, :, 0, :])
        [cx_pred, cy_pred, w_pred, h_pred, c_pred] = tf.unstack(pred_bbox, axis = -1)
        
        # Calculate each loss
        loss1 = self.coord * tf.reduce_sum(mask * ((cx_pred - cx_true) ** 2 + 
                                                   (cy_pred - cy_true) ** 2 + 
                                                   (tf.where(h_pred < 0, -tf.sqrt(-h_pred), tf.sqrt(h_pred)) - tf.sqrt(h_true)) ** 2 + 
                                                   (tf.where(w_pred < 0, -tf.sqrt(-w_pred), tf.sqrt(w_pred)) - tf.sqrt(w_true)) ** 2), 
                                           axis = (1, 2))
        loss2 = tf.reduce_sum(mask * (c_pred - tf.reduce_max(iou, axis = -1)) ** 2, axis = (1, 2))
        loss3 = self.noobj * tf.reduce_sum((tf.reduce_sum(complete_c_pred ** 2, axis = -1) - mask * c_pred ** 2), axis = (1, 2))
        loss4 = tf.reduce_sum(mask * tf.reduce_sum((pred_class - true_label) ** 2, axis = -1), axis = (1, 2))
        
        return tf.reduce_mean(loss1 + loss2 + loss3 + loss4)