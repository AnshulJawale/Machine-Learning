import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
import joblib, os
from conf import save_model, join_models

#todo Get a data frame -> Supply data to LinearRegression model -> Predict the prices

prices = [550000, 565000, 610000, 680000, 725000]               # Prices of houses per area
area = [2600, 3000, 3200, 3600, 4000]                           # Area

df = pd.DataFrame({"Area":area, "Prices":prices})               #* Making a Data Frame

#* The linear model plots the points and derives the equation of the graph -> y = mx + c
house_price_model = linear_model.LinearRegression()                           #? Creating instance of Regression Model

#* The value to be given as input should be a 2-D array
house_price_model.fit(df[["Area"]], df["Prices"])                             #? Supplying data to model

#* The model predicts the price for a given area(2-D array)
print(house_price_model.predict([[3000]]))
print(house_price_model.predict([[1000], [2300], [15000], [500]]))

print(house_price_model.coef_)                                             # Returns the slope(m) in equation y = mx + c
print(house_price_model.intercept_)                                        # Returns the y-intercept(c) in equation y = mx + c

print(df)

plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df["Area"], df["Prices"], color="red", marker="+", )
plt.plot(df["Area"], house_price_model.predict(df[["Area"]]))
plt.show()

save_model(house_price_model, "house_price_model")