import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def clusterKMeansBase(corrMatrix, maxNumClusters=10, n_init=10):
    """ Featureのクラスタリングを行う
    以下の問題を解消している;
    1. クラスタ数Kの不定性 -> silhouette score(各クラスタで定義される)のmean/stdを最大化する値を選ぶ
    2. 初期条件によるクラスタ結果の揺らぎ -> n_initを複数用意
    Args:
        corrMatrix (pd.DataFrame): Featureの相関行列
        maxNumClusters (int, optional): 最大のクラスタ数
        n_init (int, optional): initializationを何通り試すか
    Returns:
        pd.DataFrame: correlation matrix
        dict: {cluster_index: [feature_name,...]}
        pd.Series: colname -> silhscore
    """
    if maxNumClusters is None or maxNumClusters > corrMatrix.shape[1]-1:
        maxNumClusters = corrMatrix.shape[1] - 1
    distantMatrix = np.sqrt(0.5*(1 - corrMatrix))
    silh = pd.Series(dtype=float)
    stat = -np.inf
    for init in range(1, n_init+1):  # 初期化の乱数を何通りか試す
        for i in range(2, maxNumClusters+1):
            kmeans_ = KMeans(n_clusters=i, n_init=init)
            kmeans_ = kmeans_.fit(distantMatrix)
            silh_ = silhouette_samples(distantMatrix, kmeans_.labels_)
            stat_new = silh_.mean() / silh_.std()  # q in textbook
            if np.isnan(stat_new) or stat < stat_new:
                silh, kmeans, stat = silh_, kmeans_, stat_new

    # reorder columns and rows
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corrMatrix.iloc[newIdx].iloc[:, newIdx]
    # get cluster i components
    clusters = {
        i: corrMatrix.columns[np.where(kmeans.labels_ == i)[0]].tolist()
        for i in np.unique(kmeans.labels_)
    }
    silh = pd.Series(silh, index=corrMatrix.index)
    return corr1, clusters, silh


def clusterKMeansTop(corr, maxNumClusters=None, n_init=10):
    """ silhouetteスコアが悪いクラスタについて再度クラスタリングを繰り返す.
    3番目の問題; クラスタ毎の精度の不均一性を緩和している.
    Args:
        corr (pd.DataFrame): Featureの相関行列
        maxNumClusters (int, optional): 最大のクラスタ数
        n_init (int, optional): initializationを何通り試すか
    Returns:
        pd.DataFrame: correlation matrix
        dict: {cluster_index: [feature_name,...]}
        pd.Series: colname -> silhscore
    """
    if maxNumClusters is None or maxNumClusters > corr.shape[1]-1:
        maxNumClusters = corr.shape[1] - 1
    # Apply Base Clustering, then extract below mean cluster
    corr1, clusters, silh = clusterKMeansBase(corr, maxNumClusters, n_init)
    clusterTstats = {
        i: np.mean(silh[clusters[i]]) / np.std(silh[clusters[i]]) if silh[clusters[i]].shape[0] > 1 else 0
        for i in clusters.keys()
    }
    tStatMean = sum(clusterTstats.values()) / len(clusterTstats)
    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i] < tStatMean]  # list of cluster index
    if len(redoClusters) <= 1:
        return corr1, clusters, silh
    else:
        # redo clustering only in bad clusters
        keysRedo = [j for i in redoClusters for j in clusters[i]]  # list of feature names
        corrTmp = corr.loc[keysRedo, keysRedo]
        tStatMean = np.mean([clusterTstats[i] for i in redoClusters])
        corr2, clusters2, silh2 = clusterKMeansTop(  # recursive call
            corrTmp, maxNumClusters, n_init
        )
        corrNew, clustersNew, silhNew = makeNewOutputs(
            corr, {i: clusters[i] for i in clusters.keys() if i not in redoClusters}, clusters2
        )
        newTstatMean = np.mean(
            [np.mean(silhNew[clustersNew[i]]) / np.std(silhNew[clustersNew[i]]) if silhNew[clustersNew[i]].shape[0] > 1 else 0
             for i in clustersNew.keys()]
        )
        if newTstatMean <= tStatMean:
            # lower quality in new cluster, then return old one.
            return corr1, clusters, silh
        else:
            # quality improved, then return new one.
            return corrNew, corrNew, clustersNew, silhNew


def makeNewOutputs(corr, clusters, clusters2):
    """ clustersとclusters2を結合して新たにcorrとsilhスコアを計算して返す
    Args:
        corr ([type]): [description]
        clusters ([type]): [description]
        clusters2 ([type]): [description]
    Returns:
        corrNew, clustersNew, silhNew
    """
    clustersNew = {}
    for i in clusters.keys():
        clustersNew[len(clustersNew.keys())] = list(clusters[i])
    for i in clusters2.keys():
        clustersNew[len(clustersNew.keys())] = list(clusters2[i])

    newIdx = [j for i in clustersNew for j in clustersNew[i]]
    corrNew = corr.loc[newIdx, newIdx]
    x = ((1 - corr.fillna(0)) / 2)**(0.5)
    kmeans_labels = np.zeros(len(x.columns))
    for i in clustersNew.keys():
        idxs = [x.index.get_loc(k) for k in clustersNew[i]]
        kmeans_labels[idxs] = i
    silhNew = pd.Series(silhouette_samples(x, kmeans_labels), index=x.index)
    return corrNew, clustersNew, silhNew
