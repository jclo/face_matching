# ******************************************************************************
"""
Checks if the two passed-in faces are the same person.

This module implements the face recognition based on the technique
'One-Shot Learning for Face Recognition'. This technique is useful for
recognizing a person from a small reference dataset.


Private Functions:
    . _parse                    parses the script arguments,
    . _preprocess_image         returns an image array from an image path,
    . _findCosineSimilarity     returns the cosine similarity between two images,
    . _findEuclideanDistance    eeturns the euclidean distance between two images,


Public Class:
    .  FaceMatcher              a class to recognize a face against a referencial,


Public Methods:
    . compare                   compares the target face from the source face,


@namespace      -
@author         <author_name>
@since          0.0.0
@version        0.0.0
@licence        MIT. Copyright (c) 2020 Mobilabs <contact@mobilabs.fr>
"""
# ******************************************************************************
import argparse

import keras
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from models.vggface import VGGFace


WEIGHTS = './weights/vgg_face_weights.h5'
# Cosine similarity threshold:
EPSILON = 0.40


# -- Private Functions ---------------------------------------------------------

def _parse():
    """Parses the script arguments.

    ### Parameters:
        param1 ():          none.

    ### Returns:
        (str):              returns the path of the source.
        (str):              returns the path of the target.

    ### Raises:
        none
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, help='the path of the image to recognize')
    parser.add_argument('-t', '--target', type=str, help='the path of the reference image')
    args = parser.parse_args()
    return args.source, args.target


def _preprocess_image(image_path):
    """Returns an image array from an image path.

    ### Parameters:
        param1 (str):       the image path.

    ### Returns:
        (array):            returns an image array.

    ### Raises:
        none
    """
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def _findCosineSimilarity(vector_source, vector_target):
    """Returns the cosine similarity between the source and the target.

    Parameters:
        param1 (array):     the vectorized source face image.
        param2 (array):     the vectorized target face image.

    Returns:
        (num):              returns the cosine similarity.

    Raises:
        -
    """
    a = np.matmul(np.transpose(vector_source), vector_target)
    b = np.sum(np.multiply(vector_source, vector_source))
    c = np.sum(np.multiply(vector_target, vector_target))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def _findEuclideanDistance(vector_source, vector_target):
    """Returns the euclidean distance between the source and the target.

    Parameters:
        param1 (array):     the vectorized source face image.
        param2 (array):     the vectorized target face image.

    Returns:
        (num):              returns the euclidian distance.

    Raises:
        -

    """
    euclidean_distance = vector_source - vector_target
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


# -- Public --------------------------------------------------------------------

class FaceMatcher:
    """A class to recognize a face against a referencial.

    ### Attributes:
        net (obj):          the VGGFace neural network.

    ### Methods:
        compare(source, target):
            Compares the target face from the source face.

    ### Raises:
        none
    """

    def __init__(self):
        """Creates and initializes the VGGFace neural network."""
        self.net = VGGFace(include_top=False)
        self.net.load_weights(WEIGHTS)

    def compare(self, source, target):
        """Compares the target face from the source face.

        ### Parameters:
            param1 (str):   the image path of the source image.
            param2 (str):   the image path of the target image.

        ### Returns:
            (num):          the euclidean distance,
            (num):          the cosine similarity,
            (bool):         true if the two face images show the same person,

        ### Raises:
            -
        """
        vector_source = self.net.predict(_preprocess_image(source))[0, :]
        vector_target = self.net.predict(_preprocess_image(target))[0, :]

        cosine_similarity = _findCosineSimilarity(vector_source, vector_target)
        euclidean_distance = _findEuclideanDistance(vector_source, vector_target)

        if cosine_similarity < EPSILON:
            match = True
        else:
            match = False

        return euclidean_distance, cosine_similarity, match


if __name__ == '__main__':
    matcher = FaceMatcher()
    source, target = _parse()
    distance, similarity, match = matcher.compare(source, target)
    if match is True:
        m = 'Yes!'
    else:
        m = 'No!'

    print('euclidean distance: ' + str(distance)
          + ', cosine similarity: ' + str(similarity)
          + ', match: ' + m)


# -- o ---
