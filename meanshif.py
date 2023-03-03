import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == '__main__':

    dataset = pd.read_csv("data/candy.csv")
    print(dataset.head())

    X = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(X)
    print("Clusters: ",max(meanshift.labels_)+1)
    print("**"*20)
    print("----------Centers--------")
    print(meanshift.cluster_centers_)

    dataset['clusters'] = meanshift.predict(X)
    print("**"*20)

    print(dataset)