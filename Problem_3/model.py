import tensorflow as tf 

DIM_IMG = (224, 224)

class AccelerationLaw(tf.keras.layers.Layer):
    """
    Tensorflow layer to evaluate the acceleration law:

        a = g * (sin(th) - mu * cos(th))

    g is a trainable parameter because the units of acceleration in the
    dataset are pixels/frame^2, and the conversion from 9.81 m/s^2 to these
    units are unknown.
    """

    def __init__(self, **kwargs):
        super(AccelerationLaw, self).__init__(**kwargs)

    def build(self, input_shape):
        self.g = self.add_weight(name='g', shape=(1,), initializer=tf.keras.initializers.Constant(16), trainable=True)

    def call(self, inputs):
        mu, th = inputs

        ########## Your code starts here ##########
        print("========")
        print(mu)
        print(th)
        # B, M, N, C = mu.shape
        # a = [0] * 32
        # for i in range(B): 
        #     a[i] = self.g * (tf.math.sin(th[i]) - (mu[i, :,:,:] * tf.math.cos(th[i])))
        # a = tf.Variable(a)
        th_1 = tf.expand_dims(th, -1)
        th_2 = tf.expand_dims(th_1, -1)
        a = self.g * (tf.math.sin(th_2) - (tf.math.multiply(mu, tf.math.cos(th_2))))
        print("Called AccelerationLaw, calculating A")
        # print(a)
        print(a.shape)
        # a will now be the predicted acceleration
        ########## Your code ends here ##########

        # Ensure output acceleration is positive
        return a

def build_model():
    """
    Build the acceleration prediction network.

    The network takes two inputs:
        img - first frame of the video
        th  - incline angle of the ramp [rad]

    The output is:
        a - predicted acceleration of the object [pixels/frame^2]

    The last two layers of the network before the AccelerationLaw layer should be:
        p_class - A fully connected layer of size 32 with softmax output. This
                  represents a probability distribution over 32 possible classes
                  for the material of the object.
                  NOTE: Name this layer 'p_class'!
        mu - A vector of 32 weights representing the friction coefficients of
             each material class. The dot product of these weights and p_class
             represent the predicted friction coefficient of the object in the
             video.
             NOTE: Name this layer 'mu'!
    """

    img_input = tf.keras.Input(shape=(DIM_IMG[1], DIM_IMG[0], 3), name='img')
    th_input = tf.keras.Input(shape=(1,), name='th')

    ########## Your code starts here ##########
    # TODO: Create your neural network and replace the following two layers
    #       according to the given specification.

    # PB: i is one of the list of materials  
    # p_i is the probability that the material is i
    # mu_i is the friction co-eff of that material
    # mu_pred = sigma p_i u_i

    # PB: 
    # single_ds = dataset.take(1)
    # t = single_ds.get_single_element() # get a tensor from the dataset
    # >>>
    # >>> dataset.take(1).get_single_element()[0][0].shape
    # TensorShape([2, 224, 224, 3])
    # >>> dataset.take(1).get_single_element()[0][1].shape
    # TensorShape([2])
    # >>> elem[0][1]
    # <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.34906584, 0.34906584], dtype=float32)>
    # >>> dataset.take(1).get_single_element()[1].shape
    # TensorShape([2])
    # >>> elem[1]
    # <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.642273 , 1.4265637], dtype=float32)>
    # >>> len(dataset.take(1).get_single_element())
    # 2

    p_class = tf.keras.layers.Dense(32, name='p_class', activation='softmax')(img_input)
    mu = tf.keras.layers.Dense(32, name='mu')(p_class)

    ########## Your code ends here ##########

    a_pred = AccelerationLaw(name='a')((mu, th_input))

    return tf.keras.Model(inputs=[img_input, th_input], outputs=[a_pred])

def build_baseline_model():
    """
    Build a baseline acceleration prediction network.

    The network takes one input:
        img - first frame of the video

    The output is:
        a - predicted acceleration of the object [pixels/frame^2]

    The structure of this network should match the other model before the
    p_class layer. Instead of outputting p_class, it should directly output a
    scalar value representing the predicted acceleration (without using the
    AccelerationLaw layer).
    """

    img_input = tf.keras.Input(shape=(DIM_IMG[1], DIM_IMG[0], 3), name='img')
    th_input = tf.keras.Input(shape=(1,), name='th')

    ########## Your code starts here ##########
    # TODO: Replace the following with your model from build_model().

    ########## Your code ends here ##########

    return tf.keras.Model(inputs=[img_input, th_input], outputs=[a_pred])

def loss(a_actual, a_pred):
    """
    Loss function: L2 norm of the error between a_actual and a_pred.
    """

    ########## Your code starts here ##########
    # l = tf.norm(tf.expand_dims(a_actual-a_pred, -1), axis=1, ord='euclidean')
    # print("Printing shapes...")
    # print(a_actual.numpy())
    # print(a_actual[:] * 2)
    
    # for element in a_actual:
    #     print(element)
    # d = tf.Variable(a_actual)
    # print(tf.convert_to_tensor(a_actual))
    # print(a_actual.get_single_element())
    # print(d[0])
    # print(a_actual.shape)
    # print(a_actual)
    # print(a_pred.shape)
    # l = tf.multiply(a_pred, a_pred)
    # print(l)
    a1 = tf.expand_dims(a_actual, -1)
    a2 = tf.expand_dims(a1, -1)
    a3 = tf.expand_dims(a2, -1)
    l = 0.0
    i = 0
    for acceleration in a_actual:
        i += 1
        l +=  tf.norm(tf.cast(acceleration, a_pred.dtype)  - a_pred, ord='euclidean')
    # B, M, N, C = a_pred.shape
    l = l / tf.cast(i, tf.float32)
    # l = tf.norm(a_actual - a_pred, ord='euclidean')
    ########## Your code ends here ##########

    return l
