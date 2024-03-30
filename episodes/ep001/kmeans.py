import numpy as np
import plotnine as gg
import pandas as pd

from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
X = (X - X.mean(axis=0)) / X.std(axis=0)

X = pd.DataFrame(X, columns=("x1", "x2"))

X["cluster"] = "0"
print(X.head())

def get_rand_indxs(n, k):
    return(np.random.randint(0, n, size=k))

np.random.seed(42) # Fix the random seed for reproducibility
rindxs = get_rand_indxs(X.shape[0], 3)
centers = X.iloc[rindxs, 0:2]


X.iloc[rindxs, 2] = ["1", "2", "3"]

def get_clusters(X, centers):
    A = X.iloc[:, 0:2].to_numpy()
    B = centers.iloc[:, 0:2].to_numpy()
    distances = np.sqrt(((A - B[:, np.newaxis]) ** 2).sum(axis=2))
    return([["1", "2", "3"][i] for i in distances.argmin(axis=0)])

clusters = get_clusters(X.iloc[:, 0:2], centers)
X.cluster = clusters

def get_centers(X):
    return X.groupby("cluster").mean().reset_index(drop=True)

centers = get_centers(X)

while True:
    old_clusters = X.cluster
    X.cluster = get_clusters(X.iloc[:, 0:2], centers)
    if np.array_equal(X.cluster, old_clusters):
        break
    centers = get_centers(X)


ggplt = (
    gg.ggplot(data=X, mapping=gg.aes(x="x1", y="x2", color="cluster")) + 
    gg.geom_point() + 
    gg.coord_fixed() + 
    gg.theme_minimal() + 
    gg.scale_color_manual(values=["#c20305", "#377eb8", "#4daf4a"])
)

ggplt
ggplt.draw(show=True)
