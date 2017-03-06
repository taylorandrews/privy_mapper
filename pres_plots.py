import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

if __name__ == '__main__':

    df = pd.read_csv('data/21057.csv')

    X_full = df.values
    X = X_full[:, [13, 14, 16]]
    Y = X_full[:, [18]]

    x_min, x_max = -76.545, -76.45 #X[:, 1].min() - .05, X[:, 1].max() + .05
    y_min, y_max = 39.426, 39.49 #X[:, 0].min() - .05, X[:, 0].max() + .05

    font = {'fontname':'Oswald'}
    # fig, ax = plt.subplots()
    plt.gcf().subplots_adjust(bottom=0.15, left=0.4ipython)
    plt.style.use('ggplot')
    plt.figure(2, figsize=(15, 15), dpi=40)
    plt.clf()
    plt.ticklabel_format(useOffset=False)
    plt.suptitle('Agglomerative Clustering', fontsize=100, **font)
    plt.title('Zip Code: 21057', fontsize=70, **font)
    plt.scatter(X[:, 1], X[:, 0], color='black', s=500, cmap=plt.cm.Paired)
    plt.xlabel('Longitude', fontsize=70)
    plt.ylabel('Latitude', fontsize=70)
    plt.xticks(fontsize=70)
    plt.yticks(fontsize=70)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # # To getter a better understanding of interaction of the dimensions
    # # plot the first three PCA dimensions
    # fig = plt.figure(1, figsize=(8, 6))
    # ax = Axes3D(fig, elev=-150, azim=110)
    # # X_reduced = PCA(n_components=3).fit_transform(iris.data)
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y,
    #            cmap=plt.cm.Paired)
    # ax.set_title("First three PCA directions")
    # ax.set_xlabel("1st eigenvector")
    # ax.w_xaxis.set_ticklabels([])
    # ax.set_ylabel("2nd eigenvector")
    # ax.w_yaxis.set_ticklabels([])
    # ax.set_zlabel("3rd eigenvector")
    # ax.w_zaxis.set_ticklabels([])
    plt.show()
    plt.savefig('plots/agg_clustering_21057_2D.jpg')
