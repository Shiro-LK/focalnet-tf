import tensorflow as tf 
import tensorflow_probability as tfp 


class DropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob 
        self.scale_by_keep = scale_by_keep
        
    def sampling(self, x, shape, prob):
        if not shape[0]:
            return x
        random_tensor  = tf.random.uniform(shape=shape, minval=0, maxval=1 )
        random_tensor  = tf.where(random_tensor < prob, 1., 0.)
        if prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor / prob
        #print(random_tensor)
        #print("inside sampling :", x.shape, random_tensor.shape)
        return x * tf.cast(random_tensor, dtype=x.dtype)    

    def drop_path(self, x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)  # work with diff dim tensors, not just 2D ConvNets
 
        random_tensor = self.sampling(x, shape, keep_prob)
 
        return random_tensor 

    def call(self, x, training):
        return self.drop_path(x, self.drop_prob, training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'