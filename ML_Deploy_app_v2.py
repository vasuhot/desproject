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
# Features to exclude from handling of Ouliers
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


st.header('Prediction of Equipment /Machine Failure :')

# Encoding "Type" attribute
le = preprocessing.LabelEncoder()
df['Type']= le.fit_transform(df['Type'])

df_preproc = df.copy() # creating the copy
st.header("DATA")
st.write(df)

X = df.drop(["Machine failure","TWF","HDF","PWF","OSF","RNF"], axis=1)
#X = df.drop(["Machine failure","TWF","HDF","PWF","OSF","RNF"], axis=1)
# Make all failure modes into a single feature for multiclass classification.
# considered "Machine failure" with all Failure modes

# Following number is used for Machine failure modes
# 0 for No Failure
# 1 for tool wear failure (TWF)
# 2 for heat dissipation failure (HDF)
# 3 for power failure (PWF)
# 4 for overstrain failure (OSF)
# 5 for random failures (RNF)
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
    options = ["RandomForestClassifier_MC"]
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


# Building the Model
model = RandomForestClassifier(n_estimators = 100,n_jobs=-1,random_state=0,bootstrap=True).fit(X_train_mc,Y_train_mc)
# Predicting
y_predictions = model.predict(df_input)

# Predict for test data
if testdata:
    y_ptest = model.predict(X_test_mc)

# Machine Failure and  modes
st.header("Equipment/Machine failure modes")
st.write("0 for No Failure")
st.write("1 for tool wear failure (TWF)")
st.write("2 for heat dissipation failure (HDF)")
st.write("3 for power failure (PWF)")
st.write("4 for overstrain failure (OSF)")
st.write("5 for random failures (RNF)")

st.header("Prediction")
st.write(y_predictions)

if testdata:
    st.write("Prediction for Test data")
    #st.write(y_ptest)
   # uq, freq = np.unique(y_ptest, return_counts=True)
   #uf = np.asarray((uq, freq))
   # st.write('Predicted Failure Modes:', uf)

    # array to Series
    ser = pd.Series(y_ptest)
    # frequency of unique values
    count = ser.value_counts()

    for val, i in count.items():
     st.write(f"{val} Predicted {i} times")