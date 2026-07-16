# Lesson 4 — Data Preprocessing

import pandas as pd  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
df = pd.read_csv('data/salary_dataset.csv')

# print(type(df["JobTitle"][0]))
# print(type(df["Experience"][0]))
# print(type(df["Salary"][0]))

encoder = LabelEncoder() 
df["JobTitle"]= encoder.fit_transform(df["JobTitle"]) # fit_transform() method is used to fit the encoder to the column "JobTitle" and then transform the values in the column to numerical values.
# print(df)

x = df[["JobTitle", "Experience"]] # selecting the columns "JobTitle", "Experience" and "Salary" from the DataFrame df and assigning it to the variable x.
y = df["Salary"] # selecting the column "Salary" from the DataFrame df and assigning it to the variable y.

# print("\nX--->", x) # printing the variable x.
# print("\nY--->", y) # printing the variable y.
# print(encoder.classes_) # classes_ attribute returns the unique classes in the column "JobTitle" after encoding.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # train_test_split() method is used to split the data into training and testing sets. test_size=0.2 means 20% of the data will be used for testing and 80% for training. random_state=42 is used to ensure that the split is reproducible.
print("\nX_train--->", x_train.head(), "length: ",len(x_train))
print("\nX_test--->", x_test.head(), "length: ",len(x_test)) 
print("\nY_train--->", y_train.head(), "length: ",len(y_train)) 
print("\nY_test--->", y_test.head(), "length: ",len(y_test)) 
