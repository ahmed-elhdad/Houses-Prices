import pandas
from sklearn import linear_model
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split
import visualize

scale = StandardScaler()
df = pandas.read_csv("house_data.csv")
X = df[["Size", "Bedrooms", "Age"]]
y = df["Price"]
regr = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_scaled = scale.fit_transform(X)
regr.fit(X_scaled, y)
scaled = scale.transform(X)
predicted_price = regr.predict([scaled[0]])
print(predicted_price)

r2score = r2_score(y_test, predicted_price)
mae = mean_absolute_error(y_test, predicted_price)
mse = mean_squared_error(y_test, predicted_price)
print(f"R2 Score: {r2_score}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
