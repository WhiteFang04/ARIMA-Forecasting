#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

st.set_page_config(page_title="ARIMA Forecasitng App",     page_icon="🧊",  layout="wide",  initial_sidebar_state="expanded")
st.markdown("Data file should have [Date,Close,High,Low,Open] columns in Data")


uploaded_file = st.file_uploader("Upload your stock file (with Date, Open, High, Low, Close columns)",type=["xlsx","csv"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    data.columns = [col.strip() for col in data.columns]
    data.columns = [col.strip().lower() for col in data.columns]
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date',inplace=True)
    data = data.sort_values(by='date', ascending=True)
    data["returns"] = data["close"].pct_change()
    data.dropna(inplace =True)
    st.subheader("Raw data")
    st.dataframe(data.tail())
    
    def check_stationarity(data):
        result = adfuller(data)
        print('ADF statistics:',result[0])
        print('p-value:',result[1])
        if result[1] <= 0.05:
            print("The data is stationary.")
        else:
            print("The data is not stationary.")

    check_stationarity(data[["returns"]])

    model = auto_arima(data["returns"], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    st.text(model.summary())
    
    residulas = model.resid

    forecast_returns = model.predict(1)

    st.dataframe(forecast_returns)

    last_price = data['close'].iloc[-1]

    predicted_prices = [last_price * (1+forecast_returns.iloc[0])]
    for i in forecast_returns.iloc[1:]:
        predicted_prices.append(predicted_prices[-1]*(1+r))

    st.dataframe(predicted_prices)


# In[ ]:




