
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import pickle

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Airline_train_processed')

df.drop('Unnamed: 0', axis=1, inplace=True)

X = df.drop('Price', axis=1)
print(X.head())

y = df['Price']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)


# Building model
cat = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='RMSE')
cat.fit(X_train, y_train)
y_prediction = cat.predict(X_test)

# Saving model

pickle.dump(cat, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

print(y_prediction)
