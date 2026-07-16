import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import joblib

encoder = LabelEncoder()
model = LinearRegression()
df = pd.read_csv('data/salary_data_extended.csv')

# print(df.head())
df["JobTitle"]= encoder.fit_transform(df["JobTitle"])

x= df[["JobTitle", "Experience"]]
y= df["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

model.fit(x_train, y_train)

predictions = model.predict(x_test)
# print("\nPredictions--->")
# print(predictions)
# print("\nActual--->")
# print(y_test)
print("\nModel Score--->", model.score(x_test, y_test)) 
# print("\nMean Absolute Error--->", mean_absolute_error(y_test, predictions))
# print("\nMean Squared Error--->", mean_squared_error(y_test, predictions)) 
# print("\nRoot Mean Squared Error--->", root_mean_squared_error(y_test, predictions)) 
# print ("\nR2 Score--->", r2_score(y_test, predictions))

# print("\nX_train--->", x_train.head(), "length: ",len(x_train))
# print("\nX_test--->", x_test.head(), "length: ",len(x_test)) 
# print("\nY_train--->", y_train.head(), "length: ",len(y_train)) 
# print("\nY_test--->", y_test.head(), "length: ",len(y_test)) 

# print(df.head())

comparison = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": predictions
})

print(comparison)

joblib.dump(model, "models/salary_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")