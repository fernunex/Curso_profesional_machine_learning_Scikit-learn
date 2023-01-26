# Libraries
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# Kernel and PCA libraries 
from sklearn.decomposition import KernelPCA

# Linear Classifier
from sklearn.linear_model import LogisticRegression

# Preprocessing
from sklearn.preprocessing import StandardScaler # 0-1 scale
from sklearn.model_selection import train_test_split

# Starting point if run as script
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

    # Kernel function and decomposition
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    # Logistic Regression
    logistic = LogisticRegression(solver='lbfgs')
    logistic.fit(dt_train, y_train)
    print("SCORE KPCA: ", logistic.score(dt_test, y_test))