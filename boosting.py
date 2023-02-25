# General libraries
import pandas as pd

# Boosting and estimators
from sklearn.ensemble import GradientBoostingClassifier # Decision Trees
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

    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    boost_pred = boost.predict(X_test)
    print("***"*30)
    print(accuracy_score(y_test, boost_pred))