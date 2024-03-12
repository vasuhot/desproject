import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"Equipment_ds.csv")
df_ori = df.copy()


df = df.drop(["UDI","Product ID"], axis=1)
# Ouliers removal
# Features to exclude from handling of Ouliers handling
Features_excluded = ['Type', 'Machine failure','TWF','HDF','PWF','OSF','RNF']
for column in df.columns:
    if column not in Features_excluded:
        # calculate the IQR (interquartile range)
        QT1 = df[column].quantile(0.25)
        QT3 = df[column].quantile(0.75)
        IQRange = QT3 - QT1
        Outl = df[(df[column] <= (QT1 - 1.5 * IQRange)) | (df[column] >= (QT3 + 1.5 * IQRange))]
        if not Outl.empty:
            df.drop(Outl.index, inplace=True)


st.write('First ML Model to load and predict :sunglasses:')

le = preprocessing.LabelEncoder()
df['Type']= le.fit_transform(df['Type'])

df_preproc = df.copy() # creating the copy
st.header("DATA")
st.write(df)

X = df.drop(["Machine failure","TWF","HDF","PWF","OSF","RNF"], axis=1)
#X = df.drop(["Machine failure","TWF","HDF","PWF","OSF","RNF"], axis=1)
conditions = [
    (df['TWF'] == 1),
    (df['HDF'] == 1),
    (df['PWF'] == 1),
    (df['OSF'] == 1),
    (df['RNF'] == 1)
]

choices = [
    1,
    2,
    3,
    4,
    5
]

df_preproc['Machine failure'] = np.select(conditions, choices, default=0)
Y2 = df_preproc["Machine failure"]

# Single Target Varibale with SIX Classes ( 0,1,2,3,4, AND 5) - Multi Class
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=42,stratify = Y)
X_train_mc, X_test_mc, Y_train_mc, Y_test_mc = train_test_split(X,Y2,test_size=0.30,random_state=42)

model_select = st.sidebar.selectbox(
    label = "Select the Model",
    options = ["SVC", "MPl", "xgboost"]
)

st.sidebar.header('Specify Input Parameters')
testdata = st.sidebar.checkbox('Use Test data',label_visibility="visible")
# Get user input
def user_inputs():
    product_type = st.sidebar.selectbox('Product Type',('L', 'M', 'H'))
    #air_temparature = st.sidebar.slider('Air temperature [K]', 295.0,305.0,298.5,0.1)
    air_temparature = st.sidebar.slider('Air temperature [K]', float(df['Air temperature [K]'].min()),float(df['Air temperature [K]'].max()),float(df['Air temperature [K]'].mean()),0.1)
    process_temparature = st.sidebar.slider('Process temperature [K]',305.0,315.0,310.5,0.1)
    rotational_speed = st.sidebar.slider('Rotational speed [rpm]',1160,2890,1180,1)
    torque= st.sidebar.slider('Torque [Nm]',3.5,77.0, 10.5,0.1)
    tool_wear= st.sidebar.slider('Tool wear [min]',0,255, 20,1)
    ptype=0
    if product_type == 'L':
        ptype = 1
    if product_type == 'M':
        ptype = 2
    if product_type == 'H':
        ptype = 0
    data = {
        'Type': ptype,
        'Air temperature [K]': air_temparature,
        'Process temperature [K]': process_temparature,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }
    features = pd.DataFrame(data, index=[0])
    return features


df_input = user_inputs()
st.header("Specified Input Parameters")
st.write(df_input)
st.write("---")

## H=0;L=1;M=2
#reg = joblib.load(r'C:\Users\sesa138968\Vasu_Programs\BitsProj\MLPClassifier_model.joblib')
#scaler = StandardScaler()
#sample1arr = np.array([[2,2980.1, 308.6,1551,42.8,0],[1,298.8, 308.9, 1455, 41.3, 208],[2,297,308.3,1399,46.4,132],[0,300.4,311.9,1438,46.7,41],[0,3000.4,3110.9,1438,46.7,4100]])
#sample1arr = scaler.fit_transform(df_input)
#sample1arrt = Normalizer().fit_transform(df_input)


model = RandomForestClassifier(n_estimators = 100,n_jobs=-1,random_state=0,bootstrap=True).fit(X_train_mc,Y_train_mc)

y_predictions = model.predict(df_input)

if testdata:
    y_ptest = model.predict(X_test_mc)
#jpred = reg.predict(sample1arr)
#jpred_wos = reg.predict(df_input)

#st.header("Predictions-with scal")
#st.write(jpred)
st.write("---")

st.header("Following number is used for Machine failure modes")
st.write("0 for No Failure")
st.write("1 for tool wear failure (TWF)")
st.write("2 for heat dissipation failure (HDF)")
st.write("3 for power failure (PWF)")
st.write("4 for overstrain failure (OSF)")
st.write("5 for random failures (RNF)")
st.write("---")

st.header("Prediction")
st.write(y_predictions)
st.write("---")
if testdata:
    st.write(y_ptest)
    unique, frequency = np.unique(y_ptest, return_counts=True)
    # print unique values array
    #print("Machine Failure:", unique)
    # print frequency array
    #print("Frequency Values:", frequency)
    st.write('Unique Values:', unique)
    st.write('Frequency Values:', frequency)
