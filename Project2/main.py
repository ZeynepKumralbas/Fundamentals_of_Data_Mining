from preprocessing import preprocessing

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import cdist  # to compute distance between each pair of the two collections of inputs

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import time
from eclatFile import eclatFile

import sklearn.cluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

import sklearn.metrics.pairwise as sim



def main():
    prep = preprocessing()
    data_x, data_y = prep.get_data()

    # create color dictionary
    colors = {1: 'orange', 2: 'cornflowerblue', 3: 'purple'}
    visualizeData(data_x, data_y, colors, 'Original Labeled Seeds')

    evaluateClusterPerformance(data_x, data_y, KMEANS(data_x, data_y), AGNES(data_x, data_y), DBSCAN_(data_x, data_y))


    fpm_data = prep.get_FPM_data()
    minSup = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
  #  minSup = [0.20]


    APRIORI(fpm_data, minSup)
    FPGROWTH(fpm_data, minSup)
    ECLAT(minSup)

    plt.title('Performance Comparison')
    plt.xlabel('%min support')
    plt.ylabel('run time (sec)')
    plt.legend(loc='upper right')
    plt.show()


def visualizeData(data_x, data_y, color_map, title):
    # Reducing the dimensionality of the data to make it visualizable
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(data_x)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    # print(X_principal)

    # create a figure and axis
    fig, ax = plt.subplots()

    # plot each data-point
    label = ''
    for i in range(len(X_principal['P1'])):
        if color_map[data_y[i]] == 'turquoise' and label is not 'outlier':
            label = 'outlier'
            ax.scatter(X_principal['P1'].loc[i], X_principal['P2'].loc[i], color=color_map[data_y[i]], label=label)
        else:
            ax.scatter(X_principal['P1'].loc[i], X_principal['P2'].loc[i], color=color_map[data_y[i]])
    # set a title and labels
    ax.set_title(title)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    if label == 'outlier':
        ax.legend()
    plt.show()


def KMEANS(data_x, data_y):
    print("******************** K-MEANS ********************")
    elbow_method_k(data_x)
    # data_x: data to be clustered, data_y: real label of the data, kMeans_y: labels found by the k-means clustering
    # n_clusters: number of clusters
    # init='k-means++': initializes the centroids to be distant from each other, leading to better results
    # n_initint, default=10 : Number of time the k-means algorithm will be run with different centroid seeds.
    #                         The final results will be the best output of n_init consecutive runs in terms of inertia.
    # max_iterint, default=300 : Maximum number of iterations of the k - means algorithm for a single run
    # tolfloat, default=1e-4 : Convergence tolerans
    # precompute_distances {‘auto’, True, False}, default='auto’ : Precompute distances(faster but takes more memory).


    kMeans = sklearn.cluster.KMeans(n_clusters=3)  # create k-means cluster
    kMeans.fit(data_x)  # cluster the data
    kMeans_y = kMeans.labels_  # labels of the data, labels are start with 0


    colors = {0: 'purple', 1: 'cornflowerblue', 2: 'orange'}
    visualizeData(data_x, kMeans_y, colors, 'Data Visualization for K-Means')

    return kMeans_y


def AGNES(data_x, data_y):
    print("******************** AGNES ********************")

    Z = linkage(data_x, method='ward', metric='euclidean')
    dendrogram(Z)
    plt.xlabel('Number of points in node (or index of point if no parenthesis)')
    plt.ylabel('Euclidean Distances')
    plt.title('Dendrogram')
    plt.show()

    dendrogram(Z, truncate_mode='lastp', p=10)

    plt.xlabel('Number of points in node (or index of point if no parenthesis)')
    plt.ylabel('Euclidean Distances')
    plt.title('Dendrogram (Last 10 Merges)')
    plt.show()

    y = [10.2, 5, 3, 2.3, 1.8]
    x = [1, 2, 3, 4, 5]
    plt.ylabel('Distance btw Links that Clusters are Merged')
    plt.xlabel('Number of Clusters')
    plt.title('Elbow Method Showing the Termination Condition')
    plt.yticks(np.arange(0, 11, 1))
    plt.plot(x, y)
    plt.show()

    ahc = AgglomerativeClustering(n_clusters=None, distance_threshold=3, affinity='euclidean', linkage='ward')
    ahc.fit_predict(data_x)


    # create color dictionary
    colors = {2: 'orange', 1: 'cornflowerblue', 0: 'purple'}
    visualizeData(data_x, ahc.labels_, colors, 'Data Visualization for AGNES')

    return ahc.labels_

def DBSCAN_(data_x, data_y):
    print("******************** DBSCAN ********************")

    elbow_method_eps(data_x)

    # # Numpy array of all the cluster labels assigned to each data point
    db_default = DBSCAN(eps=0.26, min_samples=25, metric='euclidean').fit(data_x)

    labels = db_default.labels_

    print(labels)

    # create color dictionary
    colors = {-1: 'turquoise', 1: 'cornflowerblue', 2: 'purple', 0: 'orange'}
    visualizeData(data_x, labels, colors, 'Data Visualization for DBSCAN')

    return labels


def elbow_method_k(data_x):
    # k-means determine k
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(data_x)
        kmeanModel.fit(data_x)
        distortions.append(
            sum(np.min(cdist(data_x, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data_x.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method Showing The Optimal k')
    plt.show()


def elbow_method_eps(data_x):
    dim = len(data_x.columns)
    k = 2 * dim - 1
    minPts = 2 * dim

    kdistances = dict()

    for index, row in data_x.iterrows():
        p = np.array([row['Area'], row['Perimeter'], row['Compactness'], row['Length of kernel'],
                      row['Width of kernel'], row['Asymmetry coefficient'], row['Length of kernel groove']])

        # p = np.array([row['P1'], row['P2']])

        neighbor, dist = kdist(p, index, k, data_x)
        kdistances[index] = dist

    sorted_dist = dict(sorted(kdistances.items(), key=operator.itemgetter(1), reverse=True))

    # Plot the elbow
    plt.plot(range(210), list(sorted_dist.values()), 'bx-')
    plt.xlabel('Indexes of Points')
    plt.ylabel(str(k)+'th NN Distances')
    plt.title('The Elbow Method Showing the Optimal Epsilon')
    plt.show()

    '''
    max_graph = dict()
    i = 0
    for key, value in sorted_dist.items():
        if i < 50:
            max_graph[key] = value
        else:
            break
        i += 1

    # Plot the elbow
    plt.plot(range(50), list(max_graph.values()), 'bx-')
    plt.xlabel('points')
    plt.ylabel('distances')
    plt.title('The Elbow Method showing the optimal epsilon')
    plt.show()
    '''


def kdist(p, idx_p, k, data_x):
    distances = dict()
    for index, row in data_x.iterrows():
        if idx_p != index:
            neighbor = np.array([row['Area'], row['Perimeter'], row['Compactness'], row['Length of kernel'],
                                 row['Width of kernel'], row['Asymmetry coefficient'], row['Length of kernel groove']])
            # neighbor = np.array([row['P1'], row['P2']])

            p = p.reshape(1, 7)
            neighbor = neighbor.reshape(1, 7)
            distances[index] = sim.euclidean_distances(p, neighbor)[0][0]

    # print(distances)
    sorted_dist = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

    i = 1
    for key, value in sorted_dist.items():
        if i == k:
            return key, value
        i += 1


def evaluateClusterPerformance(data_x, data_y, kmeans_y, agnes_y, dbscan_y):
    kmeans = list()
    agnes = list()
    dbscan = list()

    kmeans.append(metrics.adjusted_rand_score(data_y, kmeans_y))
    kmeans.append(metrics.normalized_mutual_info_score(data_y, kmeans_y))
    kmeans.append((metrics.homogeneity_score(data_y, kmeans_y)))
    kmeans.append((metrics.completeness_score(data_y, kmeans_y)))

    agnes.append(metrics.adjusted_rand_score(data_y, agnes_y))
    agnes.append(metrics.normalized_mutual_info_score(data_y, agnes_y))
    agnes.append((metrics.homogeneity_score(data_y, agnes_y)))
    agnes.append((metrics.completeness_score(data_y, agnes_y)))

    dbscan.append(metrics.adjusted_rand_score(data_y, dbscan_y))
    dbscan.append(metrics.normalized_mutual_info_score(data_y, dbscan_y))
    dbscan.append((metrics.homogeneity_score(data_y, dbscan_y)))
    dbscan.append((metrics.completeness_score(data_y, dbscan_y)))

    width = 0.2
    plt.xticks([0.20, 1.2, 2.20, 3.20], ['ARI', 'NMI', 'Homogeneity', 'Completeness'])
    plt.bar(np.arange(len(kmeans)), kmeans, width=width, label="KMeans", color='cornflowerblue')
    plt.bar(np.arange(len(agnes)) + width, agnes, width=width, label="AGNES", color='purple')
    plt.bar(np.arange(len(dbscan)) + 2 * width, dbscan, width=width, label="DBSCAN", color='gold')
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
    plt.title('Performance Comparison')
    plt.show()


# **************************************** FREQUENT PATTERN MINING ****************************************

def APRIORI(dataset, minSup):
    print("******************** APRIORI ********************")
    times = []
    for i in minSup:
        runTime = 0
        for countRun in range(0, 20):
            start = time.time()
            result = apriori(dataset, min_support=i, use_colnames=True)
            end = time.time()
            runTime += end - start
        result = result.sort_values(by=['support'], ascending=False)
        pd.set_option('display.max_rows', 100)
     #   print(result)
        file = open('apriori_output_freqitems.txt', 'w+')
        file.write(result.to_string())
        runTime = runTime / 20
        times.append(runTime)


    plt.plot(minSup, times, label='APRIORI')


def FPGROWTH(dataset, minSup):
    print("******************** FP-GROWTH ********************")
    times = []

    for i in minSup:
        runTime = 0
        for countRun in range(0, 20):
            start = time.time()
            result = fpgrowth(dataset, min_support=i, use_colnames=True)
            end = time.time()
            runTime += end - start

        result = result.sort_values(by=['support'], ascending=False)
        pd.set_option('display.max_rows', 100)
    #    print(result)
        file = open('fpgrowth_output_freqitems.txt', 'w+')
        file.write(result.to_string())
        runTime = runTime / 20
        times.append(runTime)

    plt.plot(minSup, times, label='FP-Growth')


def ECLAT(minSup):
    print("******************** ECLAT ********************")
    times = []
    for i in minSup:
        runTime = 0
        for countRun in range(0, 20):
            ef = eclatFile(i, minSup)
            runTime += ef.getRunTimeList()
        runTime = runTime / 20
        times.append(runTime)

    plt.plot(minSup, times, label='ECLAT')


if __name__ == '__main__':
    main()
