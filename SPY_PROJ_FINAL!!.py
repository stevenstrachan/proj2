#IMPORTS
#%%
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import RocCurveDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
#%%

#CLEANING
#%%%
df = pd.read_csv("Trady Flow - Best Options Trade Ideas.csv")
df = df.drop(['Time', 'Orders', 'Exp'], axis = 1)

Vol = np.array([])
for value in df.iloc[:, 5]:
    if value.find('K') != -1:
        value = float(value[:-1]) * 1000
        Vol = np.append(Vol, value)
    elif value.find('M') != -1:
        value = float(value[:-1]) * 1000000
        Vol = np.append(Vol, value)
    else:
        value = float(value)
        Vol = np.append(Vol, value)
df['Vol'] = Vol

Prems = np.array([])
for value in df.iloc[:, 6]:
    if value.find('K') != -1:
        value = float(value[:-1]) * 1000
        Prems = np.append(Prems, value)
    elif value.find('M') != -1:
        value = float(value[:-1]) * 1000000
        Prems = np.append(Prems, value)
    else:
        value = float(value)
        Prems = np.append(Prems, value)
df['Prems'] = Prems

OI = np.array([])
for value in df.iloc[:, 7]:
    if value.find('K') != -1:
        value = float(value[:-1]) * 1000
        OI = np.append(OI, value)
    else:
        value = float(value)
        OI = np.append(OI, value)
df['OI'] = OI

le = preprocessing.LabelEncoder()
le.fit(df['Sym'])
df['Sym_enc'] = le.transform(df['Sym'])

CallP_enc = np.array([])
for value in df.iloc[:, 1]:
    if value == 'Call':
        value = 1
        CallP_enc = np.append(CallP_enc, value)
    else:
        value = 0
        CallP_enc = np.append(CallP_enc, value)
df['C/P_enc'] = CallP_enc

df = df[['Sym', 'C/P', 'Sym_enc', 'C/P_enc', 'Strike', 'Spot', 'BidAsk', 'Vol', 'Prems', 'OI', 'Diff(%)', 'ITM']]

X = df.iloc[:, 2:-1]                                           
y = df.ITM

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.6, random_state=13)
my_scaler = StandardScaler()     
my_scaler.fit(X_train)
X_train_scaled = my_scaler.transform(X_train) 
X_test_scaled = my_scaler.transform(X_test) 
#%%% 

#MUTING WARNINGS
#%%
st.set_option('deprecation.showPyplotGlobalUse', False)
#%%

#HEADER AND INTRODUCTION
#%%
st.markdown("<h1 style='text-align: center; color: black;'>Option Expiration Predictors</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black;'>The purpose of this project is to test various machine learning methods and their accuracy on predicting the profitability of vanilla options contracts. </h6>", unsafe_allow_html=True)
#%%

#FORMATTING TABS
#%%
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(['Data Set Preview', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'Conclusion'])


#TAB 0
#%%
with tab0:
    st.write(df.head())
    '\n'
    '\n'
    '\n'
    st.write('Sym: underlying stock of the contract')
    st.write('C/P: a call or put option')
    st.write('Sym_enc: the encoded value of each unique stock')
    st.write('C/P_enc: the encoded value for call or put')
    st.write('Strike: the strick price of the contract')
    st.write('Spot: current price of the stock at contract origin')
    st.write('BidAsk: the bidask spread of the contract')
    st.write('Vol: number of shares traded at contract origin')
    st.write('Prems: the total money spent on this contract at origin')
    st.write('OI: the total number of opened contracts at contract origin')
    st.write('Diff %: the % difference between Spot and Strike price')
    st.write('ITM: if the contract was a win or loss (0 is loss, 1 is a win)')
    '\n'
    '\n'
    '\n'
#%%

#TAB1
with tab1:
    #AdaBoostClassifier()
    #%%%
    st.markdown("<h3 style='text-align: center; color: black;'>AdaBoost Classifier</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. </h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Applying this classifier to our data, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)

    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier',  AdaBoostClassifier())
    ])
    pipe.fit(X_train_scaled, y_train)
    y_pred0 = pipe.predict(X_train_scaled)
    y_pred1 = pipe.predict(X_test_scaled)
    training_set_score0 = precision_score(y_train, y_pred0)
    test_set_score0 = precision_score(y_test, y_pred1)
    conf_mat = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(conf_mat).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Training", training_set_score0)
    col2.metric("Test", test_set_score0)
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    
    st.markdown("<h6 style='text-align: center; color: gray;'>After some optimization, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)
    parameters = {'scaler': [StandardScaler()],
        'classifier__learning_rate': [.3995, .4, .4005],
        'classifier__n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    
    grid = GridSearchCV(pipe, parameters, scoring = 'precision', cv=2).fit(X_train_scaled, y_train)
    training_set_score = grid.score(X_train_scaled, y_train)
    test_set_score = grid.score(X_test_scaled, y_test)
    
    # Access the best set of parameters
    best_params = grid.best_params_
    #print(best_params)
    # Stores the optimum model in best_pipe
    best_pipe = grid.best_estimator_
    #st.write('Best HyperParamaters:', best_pipe)
    conf_mat = confusion_matrix(y_test, y_pred1)
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test_scaled, y_test).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    train_diff = training_set_score - training_set_score0
    col1.metric("Training", training_set_score, str(train_diff))
    test_diff = test_set_score - test_set_score0
    col2.metric("Test", test_set_score, str(test_diff))
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'> Receiver Operating Characteristic Curve: </h6>", unsafe_allow_html=True)
    y_pred = best_pipe.predict(X_test_scaled)
    RocCurveDisplay.from_predictions(y_test, y_pred).plot(color = 'k', ls = '--')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Additionally, here is the full classification report: </h6>", unsafe_allow_html=True)
    x = classification_report(y_test, y_pred1, output_dict = True)
    st.table(x)
    #%%%
    
       
with tab2:
    #DecisionTreeClassifier()
    #%%%
    st.markdown("<h3 style='text-align: center; color: black;'>Decision Tree Classifier</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>Decision Trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation. </h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Applying this classifier to our data, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)

    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier',  DecisionTreeClassifier(criterion='gini', max_depth=5))
    ])
    pipe.fit(X_train_scaled, y_train)
    y_pred0 = pipe.predict(X_train_scaled)
    y_pred1 = pipe.predict(X_test_scaled)
    training_set_score0 = precision_score(y_train, y_pred0)
    test_set_score0 = precision_score(y_test, y_pred1)
    conf_mat = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(conf_mat).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Training", training_set_score0)
    col2.metric("Test", test_set_score0)
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    
    st.markdown("<h6 style='text-align: center; color: gray;'>After some optimization, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)
    parameters = {'scaler': [StandardScaler()],
    'classifier__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    grid = GridSearchCV(pipe, parameters, scoring = 'precision', cv=2).fit(X_train_scaled, y_train)
    training_set_score = grid.score(X_train_scaled, y_train)
    test_set_score = grid.score(X_test_scaled, y_test)
    
    # Access the best set of parameters
    best_params = grid.best_params_
    #print(best_params)
    # Stores the optimum model in best_pipe
    best_pipe = grid.best_estimator_
    #st.write('Best HyperParamaters:', best_pipe)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test_scaled, y_test).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    train_diff = training_set_score - training_set_score0
    col1.metric("Training", training_set_score, str(train_diff))
    test_diff = test_set_score - test_set_score0
    col2.metric("Test", test_set_score, str(test_diff))
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'> Receiver Operating Characteristic Curve: </h6>", unsafe_allow_html=True)
    y_pred = best_pipe.predict(X_test_scaled)
    RocCurveDisplay.from_predictions(y_test, y_pred).plot(color = 'k', ls = '--')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Additionally, here is the full classification report: </h6>", unsafe_allow_html=True)
    x = classification_report(y_test, y_pred, output_dict = True)
    st.table(x)
    #%%%
    
    
with tab3:
    #GaussianNB()
    #%%%
    st.markdown("<h3 style='text-align: center; color: black;'>Gaussian Naive Bayes</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>Implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian. </h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Applying this classifier to our data, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)

    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GaussianNB())
    ])
    pipe.fit(X_train_scaled, y_train)
    y_pred0 = pipe.predict(X_train_scaled)
    y_pred1 = pipe.predict(X_test_scaled)
    training_set_score0 = precision_score(y_train, y_pred0)
    test_set_score0 = precision_score(y_test, y_pred1)
    conf_mat = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(conf_mat).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Training", training_set_score0)
    col2.metric("Test", test_set_score0)
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    
    st.markdown("<h6 style='text-align: center; color: gray;'>After some optimization, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)
    parameters = {'scaler': [StandardScaler()],
    'classifier__var_smoothing': [.00000000000005, .0000000000005, .00000005, ],
    }

    grid = GridSearchCV(pipe, parameters, scoring = 'precision', cv=2).fit(X_train_scaled, y_train)
 
    training_set_score = grid.score(X_train_scaled, y_train)
    test_set_score = grid.score(X_test_scaled, y_test)
    
    # Access the best set of parameters
    best_params = grid.best_params_
    #print(best_params)
    # Stores the optimum model in best_pipe
    best_pipe = grid.best_estimator_
    #st.write('Best HyperParamaters:', best_pipe)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test_scaled, y_test).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    train_diff = training_set_score - training_set_score0
    col1.metric("Training", training_set_score, str(train_diff))
    test_diff = test_set_score - test_set_score0
    col2.metric("Test", test_set_score, str(test_diff))
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'> Receiver Operating Characteristic Curve: </h6>", unsafe_allow_html=True)
    y_pred = best_pipe.predict(X_test_scaled)
    RocCurveDisplay.from_predictions(y_test, y_pred).plot(color = 'k', ls = '--')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Additionally, here is the full classification report: </h6>", unsafe_allow_html=True)
    x = classification_report(y_test, y_pred, output_dict = True)
    st.table(x)
    #%%%
    
    
with tab4:
    #GaussianProcessClassifier()
    #%%
    st.markdown("<h3 style='text-align: center; color: black;'>Gaussian Process Classifier</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>The Gaussian Process Classifier implements Gaussian processes for classification purposes, more specifically for probabilistic classification, where test predictions take the form of class probabilities. GaussianProcessClassifier places a GP prior on a latent function, which is then squashed through a link function to obtain the probabilistic classification. </h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Applying this classifier to our data, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)

    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier',  GaussianProcessClassifier(1.0 * RBF(1.0)))
    ])
    pipe.fit(X_train_scaled, y_train)
    y_pred0 = pipe.predict(X_train_scaled)
    y_pred1 = pipe.predict(X_test_scaled)
    training_set_score0 = precision_score(y_train, y_pred0)
    test_set_score0 = precision_score(y_test, y_pred1)
    conf_mat = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(conf_mat).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Training", training_set_score0)
    col2.metric("Test", test_set_score0)
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>After some optimization, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)
    parameters = {'scaler': [StandardScaler()],
    'classifier__kernel': [1**2 * Matern(length_scale=1, nu=1.5)],
    }

    grid = GridSearchCV(pipe, parameters, scoring = 'precision', cv=2).fit(X_train_scaled, y_train)
 
    training_set_score = grid.score(X_train_scaled, y_train)
    test_set_score = grid.score(X_test_scaled, y_test)
    
    # Access the best set of parameters
    best_params = grid.best_params_
    #print(best_params)
    # Stores the optimum model in best_pipe
    best_pipe = grid.best_estimator_
    #st.write('Best HyperParamaters:', best_pipe)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test_scaled, y_test).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    train_diff = training_set_score - training_set_score0
    col1.metric("Training", training_set_score, str(train_diff))
    test_diff = test_set_score - test_set_score0
    col2.metric("Test", test_set_score, str(test_diff))
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'> Receiver Operating Characteristic Curve: </h6>", unsafe_allow_html=True)
    y_pred = best_pipe.predict(X_test_scaled)
    RocCurveDisplay.from_predictions(y_test, y_pred).plot(color = 'k', ls = '--')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Additionally, here is the full classification report: </h6>", unsafe_allow_html=True)
    x = classification_report(y_test, y_pred, output_dict = True)
    st.table(x)
    #%%
  
with tab5:
    #KNeighborsClassifier()
    #%%%
    st.markdown("<h3 style='text-align: center; color: black;'>K Neighbors Classifier</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point. </h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Applying this classifier to our data, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)

    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
    ])
    pipe.fit(X_train_scaled, y_train)
    y_pred0 = pipe.predict(X_train_scaled)
    y_pred1 = pipe.predict(X_test_scaled)
    training_set_score0 = precision_score(y_train, y_pred0)
    test_set_score0 = precision_score(y_test, y_pred1)
    conf_mat = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(conf_mat).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Training", training_set_score0)
    col2.metric("Test", test_set_score0)
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    
    st.markdown("<h6 style='text-align: center; color: gray;'>After some optimization, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)
    parameters = {'scaler': [StandardScaler(),],
        'classifier__n_neighbors': [1, 3, 5, 7, 10],
        'classifier__p': [1, 2],
        'classifier__leaf_size': [1, 5, 10, 15]
        }

    grid = GridSearchCV(pipe, parameters, scoring = 'precision', cv=2).fit(X_train_scaled, y_train)
    training_set_score = grid.score(X_train_scaled, y_train)
    test_set_score = grid.score(X_test_scaled, y_test)
    
    # Access the best set of parameters
    best_params = grid.best_params_
    #print(best_params)
    # Stores the optimum model in best_pipe
    best_pipe = grid.best_estimator_
    #st.write('Best HyperParamaters:', best_pipe)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test_scaled, y_test).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    train_diff = training_set_score - training_set_score0
    col1.metric("Training", training_set_score, str(train_diff))
    test_diff = test_set_score - test_set_score0
    col2.metric("Test", test_set_score, str(test_diff))
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'> Receiver Operating Characteristic Curve: </h6>", unsafe_allow_html=True)
    y_pred = best_pipe.predict(X_test_scaled)
    RocCurveDisplay.from_predictions(y_test, y_pred).plot(color = 'k', ls = '--')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Additionally, here is the full classification report: </h6>", unsafe_allow_html=True)
    x = classification_report(y_test, y_pred, output_dict = True)
    st.table(x)
    #%%%
    
    
with tab6:
    # MLPClassifier()
    #%%%
    st.markdown("<h3 style='text-align: center; color: black;'>Multi-Layer Perceptron Classifier</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>This model optimizes the log-loss function using LBFGS or stochastic gradient descent </h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Applying this classifier to our data, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)

    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier',  MLPClassifier(alpha=1, max_iter=1000))
    ])
    pipe.fit(X_train_scaled, y_train)
    y_pred0 = pipe.predict(X_train_scaled)
    y_pred1 = pipe.predict(X_test_scaled)
    training_set_score0 = precision_score(y_train, y_pred0)
    test_set_score0 = precision_score(y_test, y_pred1)
    conf_mat = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(conf_mat).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Training", training_set_score0)
    col2.metric("Test", test_set_score0)
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    
    st.markdown("<h6 style='text-align: center; color: gray;'>After some optimization, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)
    parameters = {'scaler': [StandardScaler()],
    'classifier__alpha': [.25, .5, .75, 1],
    'classifier__max_iter': [1000]
    }

    grid = GridSearchCV(pipe, parameters, scoring = 'precision', cv=2).fit(X_train_scaled, y_train)
    training_set_score = grid.score(X_train_scaled, y_train)
    test_set_score = grid.score(X_test_scaled, y_test)
    
    # Access the best set of parameters
    best_params = grid.best_params_
    #print(best_params)
    # Stores the optimum model in best_pipe
    best_pipe = grid.best_estimator_
    #st.write('Best HyperParamaters:', best_pipe)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test_scaled, y_test).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    train_diff = training_set_score - training_set_score0
    col1.metric("Training", training_set_score, str(train_diff))
    test_diff = test_set_score - test_set_score0
    col2.metric("Test", test_set_score, str(test_diff))
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'> Receiver Operating Characteristic Curve: </h6>", unsafe_allow_html=True)
    y_pred = best_pipe.predict(X_test_scaled)
    RocCurveDisplay.from_predictions(y_test, y_pred).plot(color = 'k', ls = '--')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Additionally, here is the full classification report: </h6>", unsafe_allow_html=True)
    x = classification_report(y_test, y_pred, output_dict = True)
    st.table(x)
    #%%%
    
with tab7:
    #QuadraticDiscriminantAnalysis()
    #%%%
    st.markdown("<h3 style='text-align: center; color: black;'>Quadratic Discriminant Analysis Classifier</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>This model optimizes the log-loss function using LBFGS or stochastic gradient descent. </h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Applying this classifier to our data, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)

    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier',   QuadraticDiscriminantAnalysis())
    ])
    pipe.fit(X_train_scaled, y_train)
    y_pred0 = pipe.predict(X_train_scaled)
    y_pred1 = pipe.predict(X_test_scaled)
    training_set_score0 = precision_score(y_train, y_pred0)
    test_set_score0 = precision_score(y_test, y_pred1)
    conf_mat = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(conf_mat).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Training", training_set_score0)
    col2.metric("Test", test_set_score0)
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    
    st.markdown("<h6 style='text-align: center; color: gray;'>After some optimization, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)
    parameters = {'scaler': [StandardScaler()],
    'classifier__reg_param': [.00000000000005, .0000000000005, .00000005, ],
    }

    grid = GridSearchCV(pipe, parameters, scoring = 'precision', cv=2).fit(X_train_scaled, y_train)
    training_set_score = grid.score(X_train_scaled, y_train)
    test_set_score = grid.score(X_test_scaled, y_test)
    
    # Access the best set of parameters
    best_params = grid.best_params_
    #print(best_params)
    # Stores the optimum model in best_pipe
    best_pipe = grid.best_estimator_
    #st.write('Best HyperParamaters:', best_pipe)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test_scaled, y_test).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    train_diff = training_set_score - training_set_score0
    col1.metric("Training", training_set_score, str(train_diff))
    test_diff = test_set_score - test_set_score0
    col2.metric("Test", test_set_score, str(test_diff))
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'> Receiver Operating Characteristic Curve: </h6>", unsafe_allow_html=True)
    y_pred = best_pipe.predict(X_test_scaled)
    RocCurveDisplay.from_predictions(y_test, y_pred).plot(color = 'k', ls = '--')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Additionally, here is the full classification report: </h6>", unsafe_allow_html=True)
    x = classification_report(y_test, y_pred, output_dict = True)
    st.table(x)
    #%%%
    

with tab8:
    #RandomForestClassifier()
    #%%%
    st.markdown("<h3 style='text-align: center; color: black;'>Random Forest Classifier</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. </h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Applying this classifier to our data, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)

    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
    ])
    pipe.fit(X_train_scaled, y_train)
    y_pred0 = pipe.predict(X_train_scaled)
    y_pred1 = pipe.predict(X_test_scaled)
    training_set_score0 = precision_score(y_train, y_pred0)
    test_set_score0 = precision_score(y_test, y_pred1)
    conf_mat = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(conf_mat).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Training", training_set_score0)
    col2.metric("Test", test_set_score0)
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    
    st.markdown("<h6 style='text-align: center; color: gray;'>After some optimization, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)
    parameters = {'scaler': [StandardScaler()],
    'classifier__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'classifier__n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    grid = GridSearchCV(pipe, parameters, scoring = 'precision', cv=2).fit(X_train_scaled, y_train)
    training_set_score = grid.score(X_train_scaled, y_train)
    test_set_score = grid.score(X_test_scaled, y_test)
    
    # Access the best set of parameters
    best_params = grid.best_params_
    #print(best_params)
    # Stores the optimum model in best_pipe
    best_pipe = grid.best_estimator_
    #st.write('Best HyperParamaters:', best_pipe)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test_scaled, y_test).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    train_diff = training_set_score - training_set_score0
    col1.metric("Training", training_set_score, str(train_diff))
    test_diff = test_set_score - test_set_score0
    col2.metric("Test", test_set_score, str(test_diff))
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'> Receiver Operating Characteristic Curve: </h6>", unsafe_allow_html=True)
    y_pred = best_pipe.predict(X_test_scaled)
    RocCurveDisplay.from_predictions(y_test, y_pred).plot(color = 'k', ls = '--')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Additionally, here is the full classification report: </h6>", unsafe_allow_html=True)
    x = classification_report(y_test, y_pred, output_dict = True)
    st.table(x)
    #%%%
    
    
with tab9:
    #Linear SVC()
    #%%
    st.markdown("<h3 style='text-align: center; color: black;'>Linear Support Vector Classifier</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>A support vector machine constructs a hyper-plane or set of hyper-planes in a high or infinite dimensional space, which can be used for classification, regression or other tasks. </h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Applying this classifier to our data, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)

    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel="linear", C=0.025))
    ])
    pipe.fit(X_train_scaled, y_train)
    y_pred0 = pipe.predict(X_train_scaled)
    y_pred1 = pipe.predict(X_test_scaled)
    training_set_score0 = precision_score(y_train, y_pred0)
    test_set_score0 = precision_score(y_test, y_pred1)
    conf_mat = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(conf_mat).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Training", training_set_score0)
    col2.metric("Test", test_set_score0)
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    
    st.markdown("<h6 style='text-align: center; color: gray;'>After some optimization, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)
    parameters = {'scaler': [StandardScaler()],
    'classifier__C': [.025,.9, .85, .95, 1],
    }

    grid = GridSearchCV(pipe, parameters, scoring = 'precision', cv=2).fit(X_train_scaled, y_train)
    training_set_score = grid.score(X_train_scaled, y_train)
    test_set_score = grid.score(X_test_scaled, y_test)
    
    # Access the best set of parameters
    best_params = grid.best_params_
    #print(best_params)
    # Stores the optimum model in best_pipe
    best_pipe = grid.best_estimator_
    #st.write('Best HyperParamaters:', best_pipe)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test_scaled, y_test).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    train_diff = training_set_score - training_set_score0
    col1.metric("Training", training_set_score, str(train_diff))
    test_diff = test_set_score - test_set_score0
    col2.metric("Test", test_set_score, str(test_diff))
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'> Receiver Operating Characteristic Curve: </h6>", unsafe_allow_html=True)
    y_pred = best_pipe.predict(X_test_scaled)
    RocCurveDisplay.from_predictions(y_test, y_pred).plot(color = 'k', ls = '--')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Additionally, here is the full classification report: </h6>", unsafe_allow_html=True)
    x = classification_report(y_test, y_pred, output_dict = True)
    st.table(x)
    #%%
    
with tab10:
    #Gamma SVC()
    #%%
    st.markdown("<h3 style='text-align: center; color: black;'>Gamma Support Vector Classifier</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>A support vector machine constructs a hyper-plane or set of hyper-planes in a high or infinite dimensional space, which can be used for classification, regression or other tasks. Gamma defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.</h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Applying this classifier to our data, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)

    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(gamma=2, C=1))
    ])
    pipe.fit(X_train_scaled, y_train)
    y_pred0 = pipe.predict(X_train_scaled)
    y_pred1 = pipe.predict(X_test_scaled)
    training_set_score0 = precision_score(y_train, y_pred0)
    test_set_score0 = precision_score(y_test, y_pred1)
    conf_mat = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(conf_mat).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Training", training_set_score0)
    col2.metric("Test", test_set_score0)
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    '\n'
    
    st.markdown("<h6 style='text-align: center; color: gray;'>After some optimization, we obtain the confusion matrix of: </h6>", unsafe_allow_html=True)
    parameters = {'scaler': [StandardScaler()],
    'classifier__gamma': [1.971, 1.972, 1.973,],
    'classifier__C': [.8399, .84, .8401],
    }

    grid = GridSearchCV(pipe, parameters, scoring = 'precision', cv=2).fit(X_train_scaled, y_train)
    training_set_score = grid.score(X_train_scaled, y_train)
    test_set_score = grid.score(X_test_scaled, y_test)
    
    # Access the best set of parameters
    best_params = grid.best_params_
    #print(best_params)
    # Stores the optimum model in best_pipe
    best_pipe = grid.best_estimator_
    #st.write('Best HyperParamaters:', best_pipe)
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test_scaled, y_test).plot(cmap = 'gist_heat_r')
    st.pyplot()
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>With training and test set precision scores of: </h6>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    train_diff = training_set_score - training_set_score0
    col1.metric("Training", training_set_score, str(train_diff))
    test_diff = test_set_score - test_set_score0
    col2.metric("Test", test_set_score, str(test_diff))
    '\n'
    '\n'
    '\n'
    st.markdown("<h6 style='text-align: center; color: gray;'>Additionally, here is the full classification report: </h6>", unsafe_allow_html=True)
    x = classification_report(y_test, y_pred, output_dict = True)
    st.table(x)
    #%%    
    
    
    
    
   
with tab11:
    #Conclusion
    #%%%
    st.markdown("<h3 style='text-align: center; color: black;'>Further Research</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black;'>The code below could be used in conjunction with np.linspace() to create a feature value grid in order to determine the feature ranges for successful options contract purchases.</h6>", unsafe_allow_html=True)
    '\n'
    '\n'
    '\n'
    code = '''def code(): 
                import itertools
                res = list(itertools.product(*all_list))
                accepted = np.array([])
                rej = 0
                for i in res:
                i = np.array([i])
                results = best_pipe.predict(i.reshape(1, -1))
                if results[0] == 1:
                    accepted = np.append(accepted, i)
                else:
                    rej += 1'''
    st.code(code, language = 'python')
    st.markdown("<h6 style='text-align: center; color: black;'>The issue with the code above is its intractability due to feature number and density of grid search points.</h6>", unsafe_allow_html=True)
    #%%%    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    