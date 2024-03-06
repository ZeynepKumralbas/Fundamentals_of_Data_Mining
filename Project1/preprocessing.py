import numpy as np
import math as math
import os
import string
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class preprocessing:

    def __init__(self, person):

        self.personPreprocessedData = []

        self.personPreprocessedData = self.openFolder(person, self.personPreprocessedData)
        self.personDataFrame = self.createDataFrame(self.personPreprocessedData)  # create the data frame

    def openFolder(self, name, personList):
        for i in range(1, 6):

            path = "signs/" + name + str(i)
            print(path)

            # r=root, d=directories, f = files
            for r, d, f in os.walk(path):
                for file in f:
                    #   print(file)
                    personList.append(self.readData(path, file))

        return personList

    def readData(self, filePath, fileName):
        f = open(filePath + "/" + fileName, 'r')
        data = [line.split(",") for line in f]
        data = np.array(data)

        # delete redundant data
        data = np.delete(data, 13, 1)  # gs2
        data = np.delete(data, 12, 1)  # gs1
        data = np.delete(data, 11, 1)  # keycode
        data = np.delete(data, 10, 1)  # litte
        data = np.delete(data, 5, 1)  # yaw
        data = np.delete(data, 4, 1)  # pitch

        newd = np.array(data[:, :8], dtype=float)  # receiver values are not included
        # newd => x y z roll thumb fore index ring
        #   print(newd)

        label = fileName.replace(".sign", "")
        label = label.replace(string.digits, "")
        label = re.sub("\d", "", label)
        #  print("label:", label)

        numOfFrames = self.calcNumOfFrames(newd)
        #  print("numOfFrames:", numOfFrames)

        distance = self.calcDistanceFeature(newd[:, :3])
        #  print("distance:", distance)

        energy = self.calcEnergyFeature(newd)
        #  print("energy:", energy)

        min_x, max_x, min_y, max_y, min_z, max_z = self.calcBoundingBox(newd)
        #  print("BoundBox", min_x, max_x, min_y, max_y, min_z, max_z)

        sTD_x = self.calcSimpleTimeDivision(newd[:, 0], numOfFrames, 5)
        sTD_y = self.calcSimpleTimeDivision(newd[:, 1], numOfFrames, 5)
        sTD_z = self.calcSimpleTimeDivision(newd[:, 2], numOfFrames, 5)
        sTD_roll = self.calcSimpleTimeDivision(newd[:, 3], numOfFrames, 5)
        sTD_thumb = self.calcSimpleTimeDivision(newd[:, 4], numOfFrames, 5)
        sTD_fore = self.calcSimpleTimeDivision(newd[:, 5], numOfFrames, 5)
        sTD_index = self.calcSimpleTimeDivision(newd[:, 6], numOfFrames, 5)
        sTD_ring = self.calcSimpleTimeDivision(newd[:, 7], numOfFrames, 5)

        return numOfFrames, distance, energy, min_x, max_x, min_y, max_y, min_z, max_z, \
               sTD_x[0], sTD_x[1], sTD_x[2], sTD_x[3], sTD_x[4], \
               sTD_y[0], sTD_y[1], sTD_y[2], sTD_y[3], sTD_y[4], \
               sTD_z[0], sTD_z[1], sTD_z[2], sTD_z[3], sTD_z[4], \
               sTD_roll[0], sTD_roll[1], sTD_roll[2], sTD_roll[3], sTD_roll[4], \
               sTD_thumb[0], sTD_thumb[1], sTD_thumb[2], sTD_thumb[3], sTD_thumb[4], \
               sTD_fore[0], sTD_fore[1], sTD_fore[2], sTD_fore[3], sTD_fore[4], \
               sTD_index[0], sTD_index[1], sTD_index[2], sTD_index[3], sTD_index[4], \
               sTD_ring[0], sTD_ring[1], sTD_ring[2], sTD_ring[3], sTD_ring[4], \
               label

    def calcNumOfFrames(self, matrix):
        return len(matrix)

    def calcDistanceFeature(self, matrix):
        n = len(matrix)  # number of frames

        # calculate x_(i)-x_(i-1), y_(i)-y_(i-1), z_(i)-z_(i-1)
        delta = np.zeros(shape=(len(matrix), len(matrix[1])))
        for i in range(1, len(matrix)):
            delta[i, :] = np.subtract(matrix[i, :], matrix[(i - 1), :])

        # calculate deltaI = sqrroot(deltaX^2 + deltaY^2 + deltaZ^2)
        deltaAll = np.zeros(shape=(len(delta), 1))
        for i in range(0, len(delta)):
            deltaAll[i, :] = math.sqrt(math.pow(delta[i, 0], 2) + math.pow(delta[i, 1], 2) + math.pow(delta[i, 2], 2))
        #     print(deltaAll)

        # sum of deltaI = distance
        distance = np.sum(deltaAll[:, 0])

        return distance

    def calcEnergyFeature(self, matrix):
        n = len(matrix)  # number of frames

        # calculate x_(i)-x_(i-1), y_(i)-y_(i-1), z_(i)-z_(i-1)
        delta = np.zeros(shape=(len(matrix), len(matrix[1])))
        for i in range(1, len(matrix)):
            delta[i, :] = np.subtract(matrix[i, :], matrix[(i - 1), :])

        # calculate deltaX_(i)-deltaX_(i-1), deltaY_(i)-deltaY_(i-1), deltaZ_(i)-deltaZ_(i-1)
        deltaSecond = np.zeros(shape=(len(delta), len(delta[1])))
        for i in range(1, len(delta)):
            deltaSecond[i, :] = np.subtract(delta[i, :], delta[i - 1, :])

        # calculate deltaI2 = sqrroot(deltaX2^2 + deltaY2^2 + deltaZ2^2)
        deltaSecondAll = np.zeros(shape=(len(deltaSecond), 1))
        for i in range(0, len(deltaSecond)):
            deltaSecondAll[i, :] = math.sqrt(
                math.pow(deltaSecond[i, 0], 2) + math.pow(deltaSecond[i, 1], 2) + math.pow(deltaSecond[i, 2], 2))

        # for energy calculation
        mult = np.zeros(shape=(len(deltaSecond), len(deltaSecond[1])))
        for i in range(2, n):
            mult[i, :] = np.multiply(deltaSecond[i, :], delta[i, :])

        energy = np.sum(mult)

        return energy

    def calcBoundingBox(self, matrix):
        min_x = np.min(matrix[:, 0])
        max_x = np.max(matrix[:, 0])
        min_y = np.min(matrix[:, 1])
        max_y = np.max(matrix[:, 1])
        min_z = np.min(matrix[:, 2])
        max_z = np.max(matrix[:, 2])

        return min_x, max_x, min_y, max_y, min_z, max_z

    def calcSimpleTimeDivision(self, column, n, d):
        s = np.zeros(shape=d)
        for i in range(0, d):
            l = math.floor(i * (n / d)) + 1
            u = math.floor((i + 1) * (n / d))
            sum = 0
            for j in range(l - 1, u):
                sum += column[j] / (u - l + 1)
            s[i] = sum

        return s

    @property
    def getPersonPreprocessedData(self):
        return self.personPreprocessedData

    def createDataFrame(self, list):
        personDataFrame = pd.DataFrame(list,
                              columns=['numOfFrames', 'distance', 'energy', 'minX', 'maxX', 'minY', 'maxY', 'minZ',
                                       'maxZ',
                                       'sTD_x0', 'sTD_x1', 'sTD_x2', 'sTD_x3', 'sTD_x4',
                                       'sTD_y0', 'sTD_y1', 'sTD_y2', 'sTD_y3', 'sTD_y4',
                                       'sTD_z0', 'sTD_z1', 'sTD_z2', 'sTD_z3', 'sTD_z4',
                                       'sTD_roll0', 'sTD_roll1', 'sTD_roll2', 'sTD_roll3', 'sTD_roll4',
                                       'sTD_thumb0', 'sTD_thumb1', 'sTD_thumb2', 'sTD_thumb3', 'sTD_thumb4',
                                       'sTD_fore0', 'sTD_fore1', 'sTD_fore2', 'sTD_fore3', 'sTD_fore4',
                                       'sTD_index0', 'sTD_index1', 'sTD_index2', 'sTD_index3', 'sTD_index4',
                                       'sTD_ring0', 'sTD_ring1', 'sTD_ring2', 'sTD_ring3', 'sTD_ring4',
                                       'label'])

        return personDataFrame

    @property
    def getPersonDataFrame(self):
        return self.personDataFrame


    '''
    def PCA(self, m_data):
        print("PCA:")
        print(m_data)
        # Machine learning systems work with integers, we need to encode these
        # string characters into ints
        encoder = LabelEncoder()

        # Now apply the transformation to all the columns:
        for col in m_data.columns:
            m_data[col] = encoder.fit_transform(m_data[col])

        X_features = m_data.iloc[:, 0:49]
        print("X_features")
        print(X_features)
  #      y_label = m_data.iloc[:, 50]

        # Scale the features
        scaler = StandardScaler()
        X_features = pd.DataFrame(scaler.fit_transform(X_features) , columns=X_features.columns)
        print("X_features")
        print(X_features)
        # Visualize
        pca = PCA()
        pca.fit_transform(X_features)
        pca_variance = pca.explained_variance_

        print("var ratio:")
        print(pca.explained_variance_ratio_)

        plt.figure(figsize=(8, 6))
        plt.bar(range(49), pca_variance, alpha=0.5, align='center', label='individual variance')
        plt.legend()
        plt.ylabel('Variance ratio')
        plt.xlabel('Principal components')
        plt.show()

        print(pca.components_)

        print("abs ")
        print(abs(pca.components_))

        print(pca.explained_variance_)


        pca2 = PCA(n_components=10)
        pca2.fit(X_features)
        x_3d = pd.DataFrame(pca2.transform(X_features))

        print(pca2.components_)
        print("PCA")
        # Dump components relations with features:
        print(pd.DataFrame(data=x_3d, columns=x_3d.columns))
        '''
