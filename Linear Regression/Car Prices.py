import pandas as pd 
from conf import join_csv, save_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder  	    #* Gives the same x-values a certain index  

car_csv = join_csv("CarPrices")
le = LabelEncoder()
df = pd.read_csv(car_csv)

df["Car Model"] = le.fit_transform(df["Car Model"])

independant_vars = df[["Car Model", "Mileage", "Age(yrs)"]]
target_var = df["Sell Price($)"].values

car_price_model = LinearRegression()

car_price_model.fit(independant_vars, target_var)
save_model(car_price_model, "car_price_model")

if __name__ == "__main__":
    name = int(input("1 -> BMW X5\n2 -> Mercedez Benz C class\n3 -> Audi A5\nSelect the number : "))
    mileage = int(input("Enter Mileage of car: "))
    age = int(input("Enter age of car : "))

    price = car_price_model.predict([[name, mileage, age]])[0]

    if 1000000 < price < 40000:
        print("The car is not available.")
    else:
        print(f"The price of car would be ${round(price)}")
