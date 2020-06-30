import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from conf import save_model, join_csv, join_models
import joblib

#* Reading CSV file
filepath = join_csv("CanadaIncome")

df = pd.read_csv(filepath)
col = "per capita income (million US$)"
df.rename(columns={" per capita income (million US$)":col})

#* Setting up the model
canada_income_model = linear_model.LinearRegression()
canada_income_model.fit(df[["Year"]], df[col])                  #* Supplying data to model

#* Make predictions for upcoming years and add them to dataframe
upcoming_years = [2017, 2018, 2019, 2020, 2021, 2022]
predictions = [canada_income_model.predict([[i]])[0] for i in upcoming_years]

#* Plot the gragh
plt.scatter(df["Year"], df[col], color="red")
plt.plot(df["Year"], canada_income_model.predict(df[["Year"]]))
plt.xlabel('Per capita Income in million US$')
plt.ylabel("Year")
plt.show()

print(df)

save_model(canada_income_model, "canada_income_model")