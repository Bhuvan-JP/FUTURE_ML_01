import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from prophet import Prophet

start_date = input("Enter start date (YYYY-MM-DD): ")  # e.g. 2023-01-01
end_date = input("Enter end date (YYYY-MM-DD): ")      # e.g. 2024-12-31
date_range = pd.date_range(start=start_date, end=end_date, freq="D")



np.random.seed(42)
x=int(input("sales"))
sales=(x
+ 0.4 * np.arange(len(date_range))                          
+ 60 * np.sin(2 * np.pi * date_range.dayofyear / 365)     
+ 25 * np.cos(2 * np.pi * date_range.dayofweek / 7)       
+ np.random.normal(0, 20, len(date_range)).astype(int)
      )

df = pd.DataFrame({
    "date": date_range,
    "company": "xyz company",  
    "sales": sales
})


df_prophet = df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]



model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)


model.fit(df_prophet)


future = model.make_future_dataframe(periods=365, freq="D")


forecast = model.predict(future)



fig1 = model.plot(forecast)
plot.title(" ABCD company - Sales Forecast (Predicting 2026)")
plot.xlabel("Date")
plot.ylabel("Sales")
plot.show()


fig2 = model.plot_components(forecast)
plot.show()


results = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
results = results.rename(columns={
    "ds": "date",
    "yhat": "predicted_sales",
    "yhat_lower": "lower_bound",
    "yhat_upper": "upper_bound"
})


final_data = df.merge(results, on="date", how="outer")

final_data.to_csv("xyz_company_sales_forecast_2026.csv", index=False)

print(" Forecast complete! Predictions for 2026 saved to xyz_company_sales_forecast_2026.csv")