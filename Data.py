import cv2
import numpy as np


class Dataset:

    def __init__(self,path):
        self.path = path
        self.actual = 1

    def get_photo(self,name,extension):
        image = cv2.imread(self.path+"\\"+name+"."+extension,0)
        image = cv2.resize(image,(96,96))
        image = np.asarray(image)
        return image

    def get_new_dataset(self,number,name,extension,value):

        dataset = []

        for i in range(number):
            current = []
            image = self.get_photo(name+"_"+str(self.actual),extension)
            self.actual += 1
            current.append(image);current.append(value)

            dataset.append(current)

        return dataset

    def add_feature_to_dataset(self,dataset,feature):

        for data in dataset:
            data[0] = feature.execute(data[0])

        return dataset
