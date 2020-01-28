import cv2
import numpy as np
import scipy
from scipy.misc import imread
import _pickle as pickle
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import extract

class Matcher(object):

    def __init__(self, pickled_db_path="features.pickle"):
        with open(pickled_db_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        features = extract.extract_features(image_path)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()


def show_img(path):
    image_query =  mpimg.imread(path['q'])
    #image_query = cv2.resize(image_query, (0, 0), None, 0.5, 0.5)

    image_result =  mpimg.imread(path['r'])
    #image_result = cv2.resize(image_result, (0, 0), None, 0.5, 0.5)


    cv2.imshow('query image', image_query)
    cv2.imshow('result images', image_result)
    cv2.waitKey()

# def measure(match_scales):
#     therashold = 0.5
#     relevant = 0
#     for val in match_scales:
#         relevant = relevant + 1;
#         if(val > therashold): break;
#
#     irrelevant = len(match_scales) - relevant;
#     Precision = relevant/(relevant + irrelevant);
#     accuracy = relevant/irrelevant;


def run():
    images_path = './image/'
    images_path_training = './image/'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 1 random images
    sample = random.sample(files, 1)

    #extract.batch_extractor(images_path_training)

    ma = Matcher('features.pickle')


    print('Query image and result')

    # imgSam = '/home/ermicho/projects/python/imageExt/images/car-1.jpg'
    #print(sample[0])
    names, match = ma.match(sample[0], topn=100)
    print(match)
    l = len(match) - 1
    img = os.path.join(images_path_training, names[0])
    #print(img)
    res = {'q' : sample[0], 'r' : img}

    show_img(res)

    # for s in sample:
    #     print('Query image ==========================================')
    #     print(s)
    #     # show_img(s)
    #     names, match = ma.match(s, topn=3)
    #     print('Result  ========================================')
    #     img = os.path.join(images_path, names[0])
    #     print(img)

        # for i in range(1):
        #     # we got cosine distance, less cosine distance between vectors
        #     # more they similar, thus we subtruct it from 1 to get match value
        #     print('Match %s' % (1 - match[i]))
        #     # show_img(os.path.join(images_path, names[i]))
        #     img = os.path.join(images_path, names[i])
        #     print(img)


run()


