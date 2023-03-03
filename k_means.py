# General libreries
import pandas as pd

# Clustering
# This is the same KMeans but with a lower resources
from sklearn.cluster import MiniBatchKMeans



if __name__ == "__main__":

    dataset = pd.read_csv('data/candy.csv')
    print(dataset.head())

    X = dataset.drop('competitorname', axis=1)
    # The winrate needs to be scaled.

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X) 
    # batch_size is the number of samples processed before the model is update.
    # Take 8 samples then solve. Take another 8 + 8 samples and then solve again.

    print("Total Centers: ", len(kmeans.cluster_centers_))
    print("**"*20)
    print(kmeans.predict(X))

    dataset['group'] = kmeans.predict(X)
    print(dataset)
