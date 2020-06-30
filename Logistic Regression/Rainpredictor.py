import pandas as pd 
from raindatascraper import df_path
from sklearn.linear_model import LogisticRegression
import datetime
from sklearn.model_selection import train_test_split
from conf import save_model

df = pd.read_csv(df_path)

df["Date"] = pd.to_datetime(df["Date"])

strftime = []

for i in df["Date"]:
    x = datetime.datetime.strptime(str(i), "%Y-%m-%d %H:%M:%S")
    strftime.append(x.timestamp())

df["Strptime"] = strftime

df = df.groupby([df["Date"]])["Rained", "Strptime"].sum().reset_index()

x_train, x_test, y_train, y_test = train_test_split(df[["Strptime"]], df["Rained"], test_size=0.2)

rain_prediction_model = LogisticRegression()
rain_prediction_model.fit(x_train, y_train)

date = input("Enter the date (yyyy-mm-dd) : ").split('-')
date = datetime.datetime(int(date[0]), int(date[1]), int(date[2]))

prediction = bool(rain_prediction_model.predict([[date.timestamp()]])[0])
print(prediction)

save_model(rain_prediction_model, "rainpredictor")