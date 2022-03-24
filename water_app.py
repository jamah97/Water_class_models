

import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

# ML pkgs
# ML pkgs
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle

def main():



	df = pd.read_csv("water_potability.csv")
	df['Sulfate'].fillna(float(df['Sulfate'].mean()), inplace=True)
	df['Trihalomethanes'].fillna(float(df['Trihalomethanes'].mean()), inplace=True)
	df['ph'].fillna(float(df['ph'].mean()), inplace=True)

	df1 = df.drop('Potability', axis=1)

	st.subheader("Model Buidling")
	st.write("About section")
	algorithm = ["K Neighbors Classifier", "Random Forest Classifier", "Logistic Regression", "Decision Tree Classifier", "SVM"]
	choice = st.selectbox("Select Algorithm",algorithm)
	testsize = st.slider('Testing size: Select percent of data that will be used as testing data.', 0, 50)



	if choice == "K Neighbors Classifier":
		iv = st.multiselect("Select predictor variables", df1.columns.to_list())
		iv = df1[iv]
		dv = df['Potability'].values
		dv2 = pd.DataFrame(data=dv)
		iv2 = pd.DataFrame(data=iv)
		if st.button("Model Performance"):
			train_x, test_x, train_y, test_y = train_test_split(iv2,dv2, test_size=testsize, random_state=1)
			KNN = KNeighborsClassifier()
			KNN.fit(train_x, train_y)
			Yhat0=KNN.predict(test_x)
			st.write('Accuracy Score:', accuracy_score(test_y, Yhat0))
			st.write(classification_report(test_y, Yhat0))
			st.write(confusion_matrix(test_y, Yhat0))
			st.write(iv2.columns)


	if choice == "Random Forest Classifier":
		iv = st.multiselect("Select predictor variables", df1.columns.to_list())
		iv = df1[iv]
		dv = df['Potability'].values
		dv2 = pd.DataFrame(data=dv)
		iv2 = pd.DataFrame(data=iv)
		if st.button("Model Performance"):
			train_x, test_x, train_y, test_y = train_test_split(iv2,dv2, test_size=testsize, random_state=1)
			RFC = RandomForestClassifier()
			RFC.fit(train_x, train_y)
			Yhat1=RFC.predict(test_x)
			st.write('Accuracy Score:', accuracy_score(test_y, Yhat1))
			st.write(classification_report(test_y, Yhat1))
			st.write(confusion_matrix(test_y, Yhat1))
			st.write(iv2.columns.to_list())


	if choice == "Logistic Regression":
		iv = st.multiselect("Select predictor variables", df1.columns.to_list())
		iv = df1[iv]
		dv = df['Potability'].values
		dv2 = pd.DataFrame(data=dv)
		iv2 = pd.DataFrame(data=iv)
		if st.button("Model Performance"):
			train_x, test_x, train_y, test_y = train_test_split(iv2,dv2, test_size=testsize, random_state=1)
			lr = LogisticRegression()
			lr.fit(train_x, train_y)
			Yhat2=lr.predict(test_x)
			st.write('Accuracy Score:', accuracy_score(test_y, Yhat2))
			st.write(classification_report(test_y, Yhat2))
			st.write(confusion_matrix(test_y, Yhat2))
			st.write(iv2.columns)


	if choice == "SVM":
		iv = st.multiselect("Select predictor variables", df1.columns.to_list())
		iv = df1[iv]
		dv = df['Potability'].values
		dv2 = pd.DataFrame(data=dv)
		iv2 = pd.DataFrame(data=iv)
		if st.button("Model Performance"):
			train_x, test_x, train_y, test_y = train_test_split(iv2,dv2, test_size=testsize, random_state=1)
			SVM = SVC()
			SVM.fit(train_x, train_y)
			Yhat4=SVM.predict(test_x)
			st.write('Accuracy Score:', accuracy_score(test_y, Yhat4))
			st.write(classification_report(test_y, Yhat4))
			st.write(confusion_matrix(test_y, Yhat4))
			st.write(iv2.columns)


	if choice == "Decision Tree Classifier":
		iv = st.multiselect("Select predictor variables", df1.columns.to_list())
		iv = df1[iv]
		dv = df['Potability'].values
		dv2 = pd.DataFrame(data=dv)
		iv2 = pd.DataFrame(data=iv)
		if st.button("Model Performance"):
			train_x, test_x, train_y, test_y = train_test_split(iv2,dv2, test_size=testsize, random_state=1)
			DTC = DecisionTreeClassifier()
			DTC.fit(train_x, train_y)
			Yhat3=DTC.predict(test_x)
			st.write('Accuracy Score:', accuracy_score(test_y, Yhat3))
			st.write(classification_report(test_y, Yhat3))
			st.write(confusion_matrix(test_y, Yhat3))
			st.write(iv2.columns)


if __name__ == '__main__':
	main()
