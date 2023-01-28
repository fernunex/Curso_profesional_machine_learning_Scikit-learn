import pandas as pd

from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor, LinearRegression
)

from sklearn.svm import SVR  # Support vectorial machine - Regressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # Importing data
    dataset = pd.read_csv('data/felicidad_corrupt.csv')
    print(dataset.tail(15))

    # Selecting columns and target
    # axis 0 --> filas | axis 1 --> columnas
    X = dataset.drop(['country', 'score', 'rank', 'high', 'low'], axis=1)
    y = dataset['score']

    # Splitting data into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3)

    estimadores = {
        'SVR' : SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC' : RANSACRegressor(), # Meta estimator --> Default: Linear Model
        'HUBER' : HuberRegressor(epsilon=1.35), # More epsilon --> less outlier
        'LinearModel' : LinearRegression()
    }

    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)
        print("="*64)
        print(name)
        print("MSE: ", mean_squared_error(y_test, predictions))

    # Data corrupted with a large deviants in y. And X is 0 in all the features. 
    # OLS and SVR fail at fitting. But RANSAC and HUBER make a great job.  