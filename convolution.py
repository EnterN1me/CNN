import math
import matplotlib.pyplot as plt
import numpy as np
import sys

Kernel_3x3 = {"contour": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
              "horizontal": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
              "vertical": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
              "sobel_horizontal": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
              "sobel_vertical": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
              "laplacien": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
              "sharpen": np.array([[0, 1, 0], [1, -5, 1], [0, 1, 0]]),
              "blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])}


def rotation_180_degree(kernel: np.ndarray) -> np.ndarray:
    # on fait une copie du noyau
    rotated = kernel.copy()

    # taille du noyau
    hauteur = kernel.shape[0]
    largeur = kernel.shape[1]

    # on balaye le noyau en le fesant tourner
    for x in range(hauteur):
        for y in range(largeur):
            rotated[x][y] = kernel[hauteur - x - 1][largeur - y - 1]

    return rotated


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # taille de l'image et du kernel
    hauteur, largeur = image.shape[0], image.shape[1]

    hauteur_kernel, largeur_kernel = kernel.shape[0], kernel.shape[1]

    # hauteur_base et largeur_base servent a savoir ou commencer
    # pour que le kernel ne dépasse pas de la matrice image
    hauteur_base = hauteur_kernel // 2  # // is floor division (ou division euclidienne)
    largeur_base = largeur_kernel // 2

    # la matrice a return, on y enleve les bord qu'on ne pourra pas calculer
    convolved = np.zeros((hauteur - 2 * hauteur_base, largeur - 2 * largeur_base))

    # On balaye la matrice image de telle sorte a ce que le kernel ne depasse pas
    for x in range(hauteur_base, hauteur - hauteur_base):
        for y in range(largeur_base, largeur - largeur_base):
            valeur = 0

            # On balaye la matrice image sur la surface du kernel actuel pour obtenir la valeur du pixel final
            for m in range(hauteur_kernel):
                for n in range(largeur_kernel):
                    valeur += kernel[m][n] * image[x - hauteur_base + m][y - largeur_base + n]

            convolved[x - hauteur_base][y - hauteur_base] = valeur

    return convolved  # la delivrance, ca y est la matrice est prete


def reLU(image: np.ndarray):
    """ REctified Linear Unit: for each pixel, set 0 if value under 0, else keep the value"""

    # taille de l'image
    hauteur = image.shape[0]
    largeur = image.shape[1]

    for i in range(hauteur):
        for j in range(largeur):
            image[i][j] = max(0, image[i][j])

    return image  # useless but need to change FeatureExtraction model to remove


def pooling(image: np.ndarray, parametre=2) -> np.ndarray:
    """rescale image by keeping 1 pixel for each parametre by parametre square,
     it keeps the highest value of this square"""

    # verifier qu'on soit bien dans une matrice numpy
    image = np.asarray(image)

    # taille de l'image et du kernel
    hauteur = image.shape[0]
    largeur = image.shape[1]

    # la matrice a return, on rajoute des rangers de 0 si on ne peut pas calculer un bords
    pooled = np.zeros((math.ceil(hauteur / parametre), math.ceil(largeur / parametre)))

    # On balaye la matrice image avec un pas egal au parametre
    for x in range(0, math.ceil(hauteur / parametre)):
        for y in range(0, math.ceil(largeur / parametre)):
            valeur = image[x * parametre][y * parametre]

            # On balaye la matrice image sur la surface du kernel actuel pour obtenir la valeur du pixel final
            for m in range(parametre):
                for n in range(parametre):
                    # si on est hors des frontiere, on prend le max des case restant, donc valeur ne change pas et il
                    # ne se passe rien
                    try:
                        valeur = max(valeur, image[x * parametre + m][y * parametre + n])
                    except IndexError:
                        pass

            pooled[x][y] = valeur

    image = pooled
    return image


def flatten(image: np.ndarray) -> list:
    """take a 2 dimensionnal array and return it in 1 dimension"""

    # verifier qu'on soit bien dans une matrice numpy
    image = np.asarray(image)

    # taille de l'image et du kernel
    hauteur = image.shape[0]
    largeur = image.shape[1]

    # le vecteur returned
    vector = []

    for x in range(hauteur):
        for y in range(largeur):
            vector.append(image[x][y])

    return vector


def between_0_1(image):
    """pass all the image pixel between 0 and 1 and keep the ratio"""

    uppest = image[0][0]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            uppest = max(uppest, image[i][j])

    return np.dot(image.copy(), 1 / uppest)


def plot_kernel(name: str):
    plt.figure(name)
    plt.imshow(Kernel_3x3[name])
    plt.axis('off')
    plt.show()
