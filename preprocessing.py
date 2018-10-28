import os

class Preprocessing(object):

    def __init__(self):
        self.flare_path = './data/flare-data/'
        self.blurry_path = './data/blurry-data/'
        self.good_path = './data/good-data/'

    def get_training_data(self, path):
        return [f for f in os.listdir(path) if os.path.isfile(path+f)]



