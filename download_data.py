import requests
import pandas as pd

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M"
    "&community=AG"
    "&longitude=86.23"
    "&latitude=27.63"
    "&start=20100101"
    "&end=20251231"
    "&format=JSON"
)

data = requests.get(url).json()
params = data["properties"]["parameter"]

df = pd.DataFrame({
    "date": params["T2M"].keys(),
    "temp": params["T2M"].values(),
    "temp_max": params["T2M_MAX"].values(),
    "temp_min": params["T2M_MIN"].values(),
    "humidity": params["RH2M"].values(),
    "rainfall": params["PRECTOTCORR"].values(),
    "wind_speed": params["WS2M"].values(),
})

df["date"] = pd.to_datetime(df["date"])
df.to_csv("jiri_weather_2010_2025.csv", index=False)
