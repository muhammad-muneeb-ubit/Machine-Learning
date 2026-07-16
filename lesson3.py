import pandas as pd # type: ignore

df = pd.read_csv('data/salary_dataset.csv')
# print("\nMain DataFrame--->", df)
# print("\ndescribe--->", df.describe()) # describe() method is used to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values.
# print("\nhead--->", df.head()) # head() method returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it.
# print("\ntail--->", df.tail()) # tail() method returns the last n rows for the object based on position.
# print("\nshape--->", df.shape) # shape() method returns a tuple representing the dimensionality of the DataFrame.
# print("\ncolumns--->", df.columns) # columns() method returns the column labels of the DataFrame.
# print("\ndtypes--->", df.dtypes) # dtypes() method returns the data types of each column in the DataFrame.
# print("\nJobTitle--->", df["JobTitle"]) # JobTitle() method returns the values of the column "JobTitle" in the DataFrame.
# print(df["JobTitle"], df["Experience"]) # specific columns
# print(df.iloc[9]) # specific row
# print(df.isnull()) # isnull() method returns a DataFrame of the same shape as df, but with True/False values indicating whether each element is null.
# print(df.isnull().sum()) # isnull() method returns a DataFrame of the same shape as df, but with True/False values indicating whether each element is null. sum() method then counts the number of null values in each column.
# print(df["JobTitle"].unique()) # unique() method returns the unique values of the column "JobTitle" in the DataFrame.
# print(df["JobTitle"].value_counts()) # value_counts() method returns a Series containing counts of unique values in the column "JobTitle" in the DataFrame.
# print(df["Salary"].max()) # max() method returns the maximum value in the column "Salary" in the DataFrame.
# print(df["Salary"].min()) # min() method returns the minimum value in the column "Salary" in the DataFrame.
# print("mean---> ",df["Salary"].mean()) # mean() method returns the mean value in the column "Salary" in the DataFrame.
# print("mode---> ",df["Salary"].mode()) # mode() method returns the mode value in the column "Salary" in the DataFrame.
# print("median---> ",df["Salary"].median()) # median() method returns the median value in the column "Salary" in the DataFrame.
# print("std---> ",df["Salary"].std()) # std() method returns the standard deviation value in the column "Salary" in the DataFrame.
# print(df.groupby("JobTitle")["Salary"].mean()) # groupby() method is used to group the data based on the column "JobTitle" and then mean() method is used to get the average salary for each job title.
# print(df.sort_values("Salary", ascending=False)) # sort_values() method is used to sort the data based on the column "Salary" in descending order.
# print(df[df["Salary"] > 100000].count()) # This line filters the DataFrame to show only the rows where the "Salary" column has values greater than 10000.
print(df[(df["Salary"] > 100000) & (df["Experience"] > 2)]) # This line filters the DataFrame to show only the rows where the "Salary" column has values greater than 10000 and the "Experience" column has values greater than 5.
