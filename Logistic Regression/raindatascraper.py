import requests
import pandas as pd 
import datetime
import os

api_key = "88433c853dc3457a98a133446201706"

now = datetime.datetime.now().day

days = 10
dates = []
rains = []
for i in range(now-6, now):
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q=Pune&dt=2020-06-{i}"

    r = requests.get(url)
    data = r.json()
    rain = data['forecast']['forecastday'][0]['day']['totalprecip_in']

    if rain < 1.0:
        rain = False
    else:
        rain = True

    date = data['forecast']['forecastday'][0]['date']
    dates.append(date)
    rains.append(int(rain))

df = pd.DataFrame({"Date":dates, "Rained":rains})
df["Date"] = pd.to_datetime(df["Date"])


BASE_DIR = os.path.dirname(__file__)
df_path = os.path.join(BASE_DIR, "CSVs", "Rain.csv")

df.to_csv(df_path, index=False)
