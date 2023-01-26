# Libraries
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# Decomposition libraries 
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

# Linear Classifier
from sklearn.linear_model import LogisticRegression

# Preprocessing
from sklearn.preprocessing import StandardScaler # 0-1 scale
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dt_heart = pd.read_csv('data/heart.csv')

    #print(dt_heart.head(5))

    # Extracting our targets
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    # Spliting our data
    X_train, X_test, y_train, y_test = train_test_split(
        dt_features, 
        dt_target,
        test_size= 0.3,
        random_state=42
        )

    # Scaling our data
    sc_x = StandardScaler().fit(X_train)
    X_train = sc_x.transform(X_train)
    X_test = sc_x.transform(X_test)

    # checking the shapes
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    
    # PCA decomposition
    pca = PCA(n_components=3)
    pca.fit(X_train)

    # IPCA decomposition for low resources
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    # Plotting the variance explained
    #plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.savefig('img/explained_ratio.jpg')

    # Training a LogisticRegression using PCA
    logistic = LogisticRegression(solver='lbfgs')
    
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE PCA:", logistic.score(dt_test, y_test))

    # Training a LogisticRegression using IPCA
    logistic_2 = LogisticRegression(solver='lbfgs')

    dt_train_2 = ipca.transform(X_train)
    dt_test_2 = ipca.transform(X_test)
    
    logistic_2.fit(dt_train_2, y_train)
    print("SCORE IPCA:", logistic_2.score(dt_test_2, y_test))