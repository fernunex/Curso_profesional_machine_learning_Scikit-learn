# Libraries
import pandas as pd
import sklearn

# Models
from sklearn.linear_model import LinearRegression

# Regularizators
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('data/felicidad.csv')
    # print(dataset.describe())

    # Features and Targets
    X = dataset[['gdp', 'family', 'lifexp', 'freedom',
                'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    # print(X.shape)
    # print(y.shape)

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Model simple training
    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    # Model + Lasso L1 training
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    # Model + Ridge L2 training
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    # Model + Elasticnet Training
    modelElastic = ElasticNet(random_state=42, alpha=0.1 ,l1_ratio=0.001).fit(X_train, y_train)
    y_predict_elastic = modelElastic.predict(X_test)

    # Metrics
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss R^2: ",linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss R^2: ",lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge Loss R^2: ", ridge_loss)

    elastic_loss = mean_squared_error(y_test, y_predict_elastic)
    print("Elastic Net Loss R^2: ", elastic_loss)

    print("*"*40)

    # Coeficients
    print("Coef Linear")
    print(modelLinear.coef_)
    print("Coef Lasso")
    print(modelLasso.coef_)
    print("Coef Ridge")
    print(modelRidge.coef_)
    print("Coef Elastic Net")
    print(modelElastic.coef_)

