import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from keras.models import save_model

# Load old pickle model (if it works locally)
with open('StockModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Save as HDF5
model.save('StockModel.h5')





st.header('Stock Market Predicton')
stock=st.text_input('Enter Stock Symbol','GooG')
start='2012-01-01'
end='2026-01-31'
data=yf.download(stock,start,end)
st.subheader('Stock Data')
st.write(data)
data_train=pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0.1))
pas_100_days=data_train.tail(100)
data_test=pd.concat([pas_100_days,data_test],ignore_index=True)
data_test_scale=scaler.fit_transform(data_test)
st.subheader('MA50')
ma_50_days=Data.Close.rolling(50).mean()
fig1=plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'g')    
plt.show()
st.ptyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days=Data.Close.rolling(100).mean()
fig1=plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r')
plt.plot(data.Close,'g')    
plt.show()
st.ptyplot(fig2)

st.subheader('Price vs MA50 vs MA100')
ma_200_days=Data.Close.rolling(200).mean()
fig1=plt.figure(figsize=(8,6))
plt.plot(ma_200_days,'r')
plt.plot(data.Close,'g')    
plt.show()
st.ptyplot(fig3)
x=[]
y=[]
for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x,y=np.array(x).np.array(y)
predict=model.predict(x)
scale=1/scaler.scale_
predict=predict*scale
y=y*scale
st.subheader('Original Price VS Prediced Price')
fig4=plt.figure(figsize=(8,6))
plt.plot(predict,'r',label='Original Price')
plt.plot(y,'g',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
plt.pyplot(fig4)