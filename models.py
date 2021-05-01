
#%%
import pandas as pd #To work with dataset
import numpy as np #Math library
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt #to plot some parameters in seaborn
from sklearn.model_selection import train_test_split
############################### importing libraries
from sklearn.model_selection import KFold, cross_val_score # to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model

from sklearn.model_selection import GridSearchCV

# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
# %%
# to feed the random state
seed = 7


#### this is nonsense i guess
def model(X_train, X_test, y_train, y_test):
    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('XGB', XGBClassifier()))

    # evaluate each model in turn
    results = []
    names = []
    scoring = 'recall'

    for name, model in models:
            kfold = KFold(n_splits=10,shuffle=True, random_state=seed)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
            
    # boxplot algorithm comparison
    fig = plt.figure(figsize=(11,6))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()






# %%



############# MODEL 1 random forest search
def random_forest(X_train, X_test, y_train, y_test,param_grid=None):

    #Seting the Hyper Parameters
    if(param_grid==None):
        param_grid = {"max_depth": [3,5, 7, 10,None],
                    "n_estimators":[3,5,10,25,50,150],
                    "max_features": [4,7,15,20]}

    #Creating the classifier
    model = RandomForestClassifier(random_state=2)

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
    grid_search.fit(X_train, y_train)

    print(grid_search.best_score_)
    print(grid_search.best_params_)

    rf = RandomForestClassifier(max_depth=None, max_features=10, n_estimators=15, random_state=2)

    #trainning with the best params
    rf.fit(X_train, y_train)

    #Testing the model 
    #Predicting using our  model
    y_pred = rf.predict(X_test)

    # Verificaar os resultados obtidos
    print(accuracy_score(y_test,y_pred))
    print("\n")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print(fbeta_score(y_test, y_pred, beta=2))

    return rf




# %%
###################MODEL 2

from sklearn.utils import resample
from sklearn.metrics import roc_curve

def gaussian_model(X_train, X_test, y_train, y_test):
    # Criando o classificador logreg
    GNB = GaussianNB()

    # Fitting with train data
    model = GNB.fit(X_train, y_train)

    # Printing the Training Score
    print("Training score data: ")
    print(model.score(X_train, y_train))

    y_pred = model.predict(X_test)

    print(accuracy_score(y_test,y_pred))
    print("\n")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print(classification_report(y_test, y_pred))

    return model


#%% verify roc curve

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

def verify_roc_curve(model,X_train, X_test, y_train, y_test):
    #Predicting proba
    y_pred_prob = model.predict_proba(X_test)[:,1]

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


    features = []
    features.append(('pca', PCA(n_components=2)))
    features.append(('select_best', SelectKBest(k=6)))
    feature_union = FeatureUnion(features)
    # create pipeline
    estimators = []
    estimators.append(('feature_union', feature_union))
    estimators.append(('logistic', GaussianNB()))
    model = Pipeline(estimators)
    # evaluate pipeline
    seed = 7
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    print(results.mean())

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(accuracy_score(y_test,y_pred))
    print("\n")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print(fbeta_score(y_test, y_pred, beta=2))

#%% implementing a pipeline of models
def pipeline(X_train, X_test, y_train, y_test):
    #Seting the Hyper Parameters
    param_test1 = {
    'max_depth':[3,5,6,10],
    'min_child_weight':[3,5,10],
    'gamma':[0.0, 0.1, 0.2, 0.3, 0.4],
    # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 10],
    'subsample':[i/100.0 for i in range(75,90,5)],
    'colsample_bytree':[i/100.0 for i in range(75,90,5)]
    }

    #Creating the classifier
    model_xg = XGBClassifier(random_state=2)

    grid_search = GridSearchCV(model_xg, param_grid=param_test1, cv=5, scoring='recall')
    grid_search.fit(X_train, y_train)