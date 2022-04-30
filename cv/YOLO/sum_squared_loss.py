import tensorflow as tf

class SumSquaredLoss(tf.keras.losses.Loss):
    def __init__(self, coord = 5, noobj = .5):
        super(SumSquaredLoss, self).__init__()
        self.name = "sum_squared_loss"
        self.lambda_coord = coord
        self.lambda_noobj = noobj

    def _neg_sqrt(self, num):
        if num < 0:
            return -1 * tf.sqrt(tf.abs(num))
        else:
            return tf.sqrt(num)

    def __call__(self, y_true, y_pred, sample_weight = None):
        y_pred = tf.cast(tf.reshape(y_pred, (tf.shape(y_pred)[0], 7, 7, 30)), tf.float64)
        loss = sum1 = sum2 = sum3 = sum4 = sum5 = tf.constant(0.0, dtype = tf.float64)
        (bbox, label, size) = y_true
        
        for i in tf.range(tf.shape(y_pred)[0]):
            for j in tf.range(size[i]):
                b, l = bbox[i, j], label[i, j]

                # center, weight, height of the bbox w.r.t. whole image
                cx_real = (b[1] + b[3]) / 2
                cy_real = (b[0] + b[2]) / 2
                w_real = b[3] - b[1]
                h_real = b[2] - b[0]

                # grid that contains the center of the bbox
                grid_x = int(cx_real * 7)
                grid_y = int(cy_real * 7)

                grid_pred = y_pred[i, grid_x, grid_y]

                grid_x = tf.cast(grid_x, tf.float64)
                grid_y = tf.cast(grid_y, tf.float64)

                # center w.r.t. the grid
                cx_real = 7 * cx_real - grid_x
                cy_real = 7 * cy_real - grid_y

                bbox_pred = tf.reshape(grid_pred[:10],[2, 5])
                label_pred = grid_pred[10:]

                real_area = w_real * h_real
                pred_area = bbox_pred[:, 2] * bbox_pred[:, 3]
                
                xsmall = tf.maximum((bbox_pred[:, 0] + grid_x) / 7 - bbox_pred[:, 2] / 2, 
                                    [(cx_real + grid_x) / 7 - w_real / 2])
                xbig = tf.minimum((bbox_pred[:, 0] + grid_x) / 7 + bbox_pred[:, 2] / 2, 
                                  [(cx_real + grid_x) / 7 + w_real / 2])
                ysmall = tf.maximum((bbox_pred[:, 1] + grid_y) / 7 - bbox_pred[:, 3] / 2, 
                                    [(cy_real + grid_y) / 7 - h_real / 2])
                ybig = tf.minimum((bbox_pred[:, 1] + grid_y) / 7 + bbox_pred[:, 3] / 2, 
                                  [(cy_real + grid_y) / 7 + h_real / 2])
                pred_area = bbox_pred[:, 2] * bbox_pred[:, 3]
                intersect = tf.maximum(tf.constant(0, dtype = tf.float64), xbig - xsmall) * \
                tf.maximum(tf.constant(0, dtype = tf.float64), ybig - ysmall)

                iou = intersect / (pred_area + real_area - intersect)
                
                max_idx = tf.argmax(iou)

                sum1 += (bbox_pred[max_idx, 0] - cx_real) ** 2 + (bbox_pred[max_idx, 1] - cy_real) ** 2
                sum2 += (self._neg_sqrt(bbox_pred[max_idx, 2]) - tf.sqrt(w_real)) ** 2 + \
                        (self._neg_sqrt(bbox_pred[max_idx, 3]) - tf.sqrt(h_real)) ** 2
                sum3 += (bbox_pred[max_idx, 4] - label_pred[l] * iou[max_idx]) ** 2
                sum4 -= bbox_pred[max_idx, 4] ** 2
                sum5 += tf.reduce_sum(label_pred ** 2) - label_pred[l] ** 2 + (1 - label_pred[l]) ** 2

            sum4 = tf.reduce_sum(y_pred[i, :, :, 4] ** 2, axis = [0, 1]) + tf.reduce_sum(y_pred[i, :, :, 9] ** 2, axis = [0, 1])

            loss += self.lambda_coord * sum1 + self.lambda_coord * sum2 + sum3 + self.lambda_noobj * sum4 + sum5

        return loss / tf.cast(tf.shape(y_pred)[0], tf.float64)