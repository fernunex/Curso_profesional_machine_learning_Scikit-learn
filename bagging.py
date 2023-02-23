# General libraries
import pandas as pd

# Bagging and estimators
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Data Prep
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    # Importing Data
    dt_heart = pd.read_csv('data/heart.csv')
    #print(dt_heart['target'].describe())

    # Target and Data
    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    # Classifier: KNeighborsClassifier only
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_predict = knn_class.predict(X_test)

    print("Classifier Only")
    print(accuracy_score(y_test, knn_predict))

    # Classifier: KNeighborsClassifier + ensamble

    bag_class = BaggingClassifier(
        estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    
    bag_pred = bag_class.predict(X_test)

    print("Clasiffier + ensamble bagging")
    print(accuracy_score(y_test, bag_pred))    

    # Classifier: LogisticRegression only
    logit_class = LogisticRegression(max_iter=10000).fit(X_train, y_train)
    logit_predict = logit_class.predict(X_test)

    print("Classifier LogisticRegression Only")
    print(accuracy_score(y_test, logit_predict))    

    # Classifier: LogisticRegression + ensamble
    bag_class_2 = BaggingClassifier(
        estimator=LogisticRegression(max_iter=10000), n_estimators=50).fit(X_train, y_train)
    
    bag_pred2 = bag_class_2.predict(X_test)

    print("Clasiffier LogisticRegression + ensamble bagging")
    print(accuracy_score(y_test, bag_pred2)) 