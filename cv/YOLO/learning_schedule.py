import tensorflow as tf

class LearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step):
        lr = self._tf_call(step)
        return lr
    
    @tf.function
    def _tf_call(self, step):
        if step / batch_size < 30:
#             return 1e-3 + 1e-2 / 30 * step / batch_size
            return 1e-3
        
        if step / batch_size < 75:
            return 1e-4
        
        if step / batch_size < 105:
            return 1e-5
        
        return 1e-6