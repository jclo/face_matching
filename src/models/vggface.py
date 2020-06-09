# ******************************************************************************
"""
Creates and Returns the VGGFace Convolutional Neural Network.

-

Public Functions:
    . VGGFace                       return the VGGFace CNN,


@namespace      -
@author         Mobilabs
@since          0.0.0
@version        0.0.0
@licence        MIT. Copyright (c) 2020 Mobilabs <contact@mobilabs.fr>
"""
# ******************************************************************************
import keras
from keras.layers import Activation, Dropout, Flatten, Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D


# -- Public Functions ----------------------------------------------------------

def VGGFace(input_shape=(224, 224, 3), n_classes=10, include_top=True):
    """Return the VGGFace Convolutional Neural Network.

    ### Parameters:
        param1 (tuple):     width, height an depth of the input image,
        param2 (int):       number of output classes,
        param3 (bool)       include or not the prediction layer,

    ### Returns:
        (obj):              return VGGFace Convolutional Neural Network,

    ### Raises:
        none
    """
    # Create the Tensor
    input = Input(shape=input_shape)

    # Block 1
    # 1st Convolutional Layer
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='block1_conv1')(input)
    x = Activation('relu', name='block1_relu1')(x)

    # 2nd Convolutional Layer
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='block1_conv2')(x)
    x = Activation('relu', name='block1_relu2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    # 3rd Convolutional Layer
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='block2_conv1')(x)
    x = Activation('relu', name='block2_relu1')(x)

    # 4th Convolutional Layer
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='block2_conv2')(x)
    x = Activation('relu', name='block2_relu2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    # 5th Convolutional Layer
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='block3_conv1')(x)
    x = Activation('relu', name='block3_relu1')(x)

    # 6th Convolutional Layer
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='block3_conv2')(x)
    x = Activation('relu', name='block3_relu2')(x)

    # 7th Convolutional Layer
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='block3_conv3')(x)
    x = Activation('relu', name='block3_relu3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    # 8th Convolutional Layer
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block4_conv1')(x)
    x = Activation('relu', name='block4_relu1')(x)

    # 9th Convolutional Layer
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block4_conv2')(x)
    x = Activation('relu', name='block4_relu2')(x)

    # 10th Convolutional Layer
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block4_conv3')(x)
    x = Activation('relu', name='block4_relu3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # 11th Convolutional Layer
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block5_conv1')(x)
    x = Activation('relu', name='block5_relu1')(x)

    # 12th Convolutional Layer
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block5_conv2')(x)
    x = Activation('relu', name='block5_relu2')(x)

    # 13th Convolutional Layer
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='block5_conv3')(x)
    x = Activation('relu', name='block5_relu3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')(x)

    # Block 6
    # 14th Convulation Layer
    x = Conv2D(4096, (7, 7), strides=(1, 1), name='fc1_conv1')(x)
    x = Activation('relu', name='fc1_relu1')(x)
    x = Dropout(0.5)(x)

    # 15th Convulation Layer
    x = Conv2D(4096, (1, 1), strides=(1, 1), name='fc2_conv1')(x)
    x = Activation('relu', name='fc2_relu2')(x)
    x = Dropout(0.5, name='fc2_dropout')(x)

    # 16th Convulation Layer
    x = Conv2D(2622, (1, 1), strides=(1, 1), name='fc3_conv1')(x)
    x = Flatten(name='fc3_flatten')(x)

    if include_top:
        # Output Layer
        x = Activation('softmax', name='predictions_softmax')(x)

    # Create model
    model = keras.models.Model(input, x, name='vggface')
    return model


if __name__ == '__main__':
    model = VGGFace()
    model.summary()

    keras.utils.plot_model(model,
                           to_file='./diagrams/VGGFace.png',
                           show_shapes=True,
                           show_layer_names=True)

# -- o ---
