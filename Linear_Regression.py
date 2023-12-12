import warnings
warnings.filterwarnings ('ignore')

import streamlit as st
import pandas as pd
import numpy as np
from math import pi
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.markdown("# Regresión lineal")

uploaded_file = st.file_uploader('Base de datos con información a evaluar', '.xlsx')

df = pd.read_excel(uploaded_file, sheet_name='BBDD')

st.write('Archivo subido')
st.write(df)

st.write('Descripción variables')
st.write(df.describe())

st.write('Tipología variables')
st.write(df.dtypes)

df2 = df[['Longitud', 'IMD22', 'Pesats22', 'Moto22', 'IFA', 'IMD_feiner', 'IMD_dissabte', 'IMD_diumenge', 'Hora_punta', 'Volum_punta',
'V_mitjana', 'IA']]
#df3 = df[['Longitud', 'CTR', 'Macrotextura', 'IMD', 'Accidents']]

st.pyplot(sns.pairplot(data=df2, kind='kde'))

st.pyplot(sns.pairplot(data=df2))

st.write('Procediment antic')

p4 = sns.pairplot(df2, hue='IFA')

st.pyplot(p4)

#Mirar comparaciones entre dos variables en concreto para entender mejor si tienen correlación o no
p3 = sns.pairplot(df2, x_vars=['IMD22', 'Volum_punta'], y_vars = ['IA'], hue = 'V_mitjana')
st.pyplot(p3)

st.write(df2.corr())
p1 = sns.heatmap(df2.corr(),annot=True,lw=1)
i1 = p1.get_figure()
st.pyplot(i1)

#p2 = sns.boxplot(y='Accidents',x='Tipus',data=df)
#i2 = p2.get_figure()
#st.pyplot(i2)

df = df.dropna(axis='index', how='any')

X = df[['Longitud', 'IMD22', 'Pesats22', 'Moto22', 'IFA', 'IMD_feiner', 'IMD_dissabte', 'IMD_diumenge', 'Hora_punta', 'Volum_punta',
'V_mitjana']]
#X = pd.get_dummies(data=X, drop_first=True, dtype = float)
st.write(X.head())

st.write(X.dtypes)
st.write(X.describe())

Y = df[['IA']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

coeff_parameter = pd.DataFrame(model.coef_.reshape(-1,1),X.columns,columns=['Coefficient'])

st.write('Coefficient parameters')
st.write(coeff_parameter)

predictions = model.predict(X_test)

st.write('Predictions')
st.write(predictions)

#p5 = sns.regplot(y_test, x_estimator = predictions)
#i5 = p5.get_figure()

#st.pyplot(i5)

#p8 = sns.regplot(x = y_test, y = predictions)

#st.pyplot(p8.get_image())

import statsmodels.api as sm
X_train_Sm = sm.add_constant(X_train)
ls = sm.OLS(y_train,X_train_Sm).fit()
st.write(ls.summary())

st.write('Coeficientes variables')
st.write(model.coef_)
st.write('Ordenada al origen')
st.write(model.intercept_)