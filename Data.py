import os
import sys

import cv2
import numpy as np

path_to_dataset = fr"C:\Users\Micka\Documents\CNN\Dataset"


def data_gestion():
    for plant in ["Alstonia Scholaris (P2)", "Arjun (P1)", "Bael (P4)", "Basil (P8)", "Chinar (P11)", "Gauva (P3)",
                  "Jamun (P5)", "Jatropha (P6)", "Lemon (P10)", "Mango (P0)", "Pomegranate (P9)",
                  "Pongamia Pinnata (P7)"]:

        for state in ["diseased", "healthy"]:
            try:
                os.chdir(path_to_dataset + fr"\plant\{plant}\{state}")
                for name in os.listdir():
                    image = cv2.imread(name)
                    image = cv2.resize(image, (256, 256))
                    cv2.imwrite(name, image)
            except FileNotFoundError as err:
                print(err, file=sys.stderr)
            print(plant, state, "finish")


def dataset_tolist():
    result = []

    for plant in ["Alstonia Scholaris (P2)", "Arjun (P1)", "Bael (P4)", "Basil (P8)", "Chinar (P11)", "Gauva (P3)",
                  "Jamun (P5)", "Jatropha (P6)", "Lemon (P10)", "Mango (P0)", "Pomegranate (P9)",
                  "Pongamia Pinnata (P7)"]:

        for state in ["diseased", "healthy"]:
            try:
                os.chdir(path_to_dataset + fr"\plant\{plant}\{state}")
                for name in os.listdir():
                    image = cv2.imread(name, 0)
                    image = cv2.resize(image, (96, 96))
                    result.append([image, (1 if state == "healthy" else -1)])
            except FileNotFoundError as err:
                print(err, file=sys.stderr)
            print(plant, state, "finish")

    return result


# data_gestion()

def pizza_data_tolist():
    result = []

    for current in ["Pizza", "Avion"]:
        os.chdir(path_to_dataset + fr"\96\{current}")
        for name in os.listdir():
            image = cv2.imread(name, 0)
            image = cv2.resize(image, (96, 96))
            result.append([image, (1 if current == "Pizza" else -1)])

    return result
