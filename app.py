import streamlit as st
import pickle
import numpy as np

df=pickle.load(open('/content/df.pkl','rb'))
pipe = pickle.load(open('/content/pipe.pkl','rb'))
st.title("Laptop price predictor App")
st.text("This app is created using the data available on kaggle.")
st.text("The prediction here is based on the data available, hence might not align with the real-life data")

company = st.selectbox("Manufacturing Company",df['Company'].unique(),index=4)
typename = st.selectbox("Type of laptop",df['TypeName'].unique(),index=1)
cpu = st.selectbox("Processor",df['Cpu'].unique(),index=0)
ram = st.radio("RAM(in GB)",[4,6,8,12,16,24,32,64,128],index=2,horizontal=True)
gpu = st.selectbox("Graphics card",df['Gpu'].unique(),index=1)
os = st.selectbox("Operating System",df['OpSys'].unique(),index=2)
weight = st.slider("Weight(in kg)",min_value=0.7,max_value=4.7,value=2.0,step=0.1)
ips = st.selectbox("IPS Display",["Yes","No"])
touchscreen = st.radio("Touchscreen",["Yes","No"],horizontal=True)
cpu_speed = st.slider("CPU Speed",min_value=0.9,max_value=3.6,value=2.5,step=0.1)
hdd = st.radio("Hard drice capacity(in GB)[If SSD, set HDD value to 0]",[0,128,500,1000,2000],horizontal=True)
ssd = st.radio("SSD capacity(in GB)",[0,8,16,32,64,128,256,512,1000,2000],horizontal=True,index=7)
screen_size = st.slider("Screen size(measured in inches, calculated diagonally)",min_value=10.0,max_value=18.5,value=15.6,step=0.1)
screen_res = st.selectbox("Screen resolution",["2560x1600","1440x900","1920x1080","2880x1800","1366x768",
                                               "2304x1440","3200x1800","1920x1200","2256x1504","3840x2160",
                                               "2160x1440","2560x1440","1600x900","2736x1824","2400x1600"],index=3)

if st.button("PREDICT PRICE"):
  ppi = None
  if (ips=='Yes'):
    ips=1
  else:
    ips=0
  if (touchscreen=='Yes'):
    touchscreen=1
  else:
    touchscreen=0
  X_res = int(screen_res.split('x')[0])
  Y_res = int(screen_res.split('x')[1])
  ppi = (((X_res)**2)+((Y_res)**2))**0.5/screen_size

  query = np.array([[company,typename,cpu,ram,gpu,os,weight,ips,touchscreen,cpu_speed,hdd,ssd,ppi]])
  op = pipe.predict(query)
  f_op = int(round(op[0],-2))
  st.subheader("The estimated price of the laptop with the above selected configuration is ₹"+str(f_op))
