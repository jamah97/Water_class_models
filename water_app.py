

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

	st.subheader("Model Building")
	st.write("About section: Below is an ML interactive ML model builder. The objective is to assess the safety of drinking water based on 9 characteristics (pH value, Hardness, Solids (Total dissolved solids - TDS), Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, and Turbidity) to see if the water is safe for human consumption.")
	st.write("You will be able to select from any of the 9 variables as predictors along with the classifier algorithm and the train test split. By default the independent variable is potability. potability of 1 means water is safe to drink and 0 means water isn't safe to drink. After clicking on model performance, the application will generate the accuracy of the model, classification report, and confusion matrix. ")
	st.write("Datasource: https://www.kaggle.com/adityakadiwal/water-potability")
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
