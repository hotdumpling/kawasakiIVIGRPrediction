import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
import math
import random
from sklearn.model_selection import KFold
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc, roc_curve, accuracy_score, recall_score, precision_score
import lightgbm as lgb
import shap

def get_evaluation_res(y_actual, y_hat, y_hat_proba):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    
    fpr, tpr, thresholds = roc_curve(y_actual, y_hat_proba)
    AUC = auc(fpr, tpr)
    accuracy = accuracy_score(y_actual, y_hat)
           
    return fpr, tpr, AUC, accuracy, sensitivity, specificity

def drawShap(regr, x_train, colnames):
    shap.initjs()
    explainer = shap.TreeExplainer(regr)
    shap_values = explainer.shap_values(x_train)
    shap.force_plot(explainer.expected_value, shap_values[0,:],colnames,matplotlib=True)     
    shap.summary_plot(shap_values, colnames)
    shap.summary_plot(shap_values, x_train, colnames,plot_type="bar")
    shap_interaction_values = explainer.shap_interaction_values(x_train)
    shap.summary_plot(shap_interaction_values, x_train)
    #shap.force_plot(explainer.expected_value, shap_values,colnames)
    return 

random.seed(507)
np.random.seed(507)

dataset = pd.read_excel('totalSet.xlsx', index_col = 0)
dataset.dropna(axis=0, how='any', inplace=True)
colnames = [d for d in dataset.columns if d != 'IVIG resistance']
x_train = dataset.loc[dataset.index < pd.Timestamp(2018,9,1), colnames].values
y_train = dataset.loc[dataset.index < pd.Timestamp(2018,9,1), 'IVIG resistance'].values
x_test = dataset.loc[dataset.index >= pd.Timestamp(2018,9,1), colnames].values
y_test = dataset.loc[dataset.index >= pd.Timestamp(2018,9,1), 'IVIG resistance'].values


scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


val_index = np.random.randint(len(x_train), size=round(len(x_train)/4))
train_index = [i for i in range(len(x_train)) if i not in val_index]
x_val = x_train[val_index, :]
y_val = y_train[val_index]

x_train = x_train[train_index, :]
y_train = y_train[train_index] # negative: 0.8 positive: 0.2


mtds = ["logit_l1", "logit_l2", "DT", "RF", "AdaBoost", "GBM", "lightGBM"]

performance_table = pd.DataFrame(columns = ['model','AUC','accuracy','sensitivity','specificity','best hyperparameter'])

hyperparameter_logistic = [0.5, 1, 1.5, 2, 2.5, 3]
hyperparameter_DT_depth = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
hyperparameter_n_estimator = [16, 32, 64, 128]
#    hyperparameter_lightGBM_leave = [5,10,20,50,100]

shap.initjs()
plots = []
for mtd in mtds:
    print(mtd)
    if mtd == "logit_l1": # around 8 mins for all folds
        regr_list = []
        hyperparam_list = hyperparameter_logistic
        for c in hyperparam_list:
            regr_list.append(LogisticRegression(penalty='l1', C=c, class_weight = {0: 0.2,1:0.8}))#, solver='saga', max_iter=10000))
    if mtd == "logit_l2": # around 4 mins for all folds
        regr_list = []
        hyperparam_list = hyperparameter_logistic
        for c in hyperparam_list:
            regr_list.append(LogisticRegression(penalty='l2', C=c, solver='sag',max_iter=2000, class_weight = {0: 0.2,1:0.8}))
    if mtd == "DT":
        regr_list = []
        hyperparam_list = hyperparameter_DT_depth
        for c in hyperparam_list:
            regr_list.append(DecisionTreeClassifier(max_depth = c, class_weight = {0: 0.2,1:0.8}))
    if mtd == "RF":
        regr_list = []
        hyperparam_list = hyperparameter_n_estimator
        for c in hyperparam_list:
            regr_list.append(RandomForestClassifier(n_estimators=c, class_weight = {0: 0.2,1:0.8}))
    if mtd == "AdaBoost":
        regr_list = []
        hyperparam_list = hyperparameter_n_estimator
        for c in hyperparam_list:
            regr_list.append(AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1, class_weight = {0: 0.2,1:0.8}), \
                                                n_estimators=c))
    if mtd == "GBM":
        regr_list = []
        hyperparam_list = hyperparameter_n_estimator
        for c in hyperparam_list:
            regr_list.append(GradientBoostingClassifier(n_estimators=c)) # no class weight function in Gradient boosting classifier


# =============================================================================
#             Find Hyper-parameters
# =============================================================================
    AUC_val_list = []
    if mtd != "lightGBM":
        for idx, regr in enumerate(regr_list):
            regr.fit(x_train, y_train)
            y_pred = regr.predict(x_val)
            y_pred_proba = regr.predict_proba(x_val)[:,1]
            fpr, tpr, AUC, accuracy, sensitivity, specificity = get_evaluation_res(y_val, y_pred, y_pred_proba)
            AUC_val_list.append(AUC)
            print("Current model: %s, current hyper-parameter: %s, current AUC score: %.8f" % (mtd, str(hyperparam_list[idx]), AUC) )
        # Choose the optimal regr
        print("Best hyper-parameter: %s" % str(hyperparam_list[np.argmax(AUC_val_list)]))
        regr = regr_list[np.argmax(AUC_val_list)]
        best_hyperparameter = hyperparam_list[np.argmax(AUC_val_list)]
    if mtd == "lightGBM":
        hyperparam_list = hyperparameter_DT_depth
        for idx, c in enumerate(hyperparam_list):
            params = {}
            evals_result = {} 
            params['learning_rate'] = 0.02
            params['boosting_type'] = 'gbdt'
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
            params['num_leaves'] = 11
            params['min_data'] = 1
            params['max_depth'] = c
            #dealing with imbalance data
            params['pos_bagging_fraction'] = 0.8
            params['neg_bagging_fraction'] = 0.2
            params['bagging_freq'] = 2
            params['bagging_seed'] = 40

            d_train = lgb.Dataset(x_train, label=y_train)

            clf = lgb.train(params, d_train, num_boost_round = 100, evals_result=evals_result)
            #Prediction
            y_pred_proba = clf.predict(x_val)
            #convert into binary values
            y_pred = (y_pred_proba > .5).astype(int)
            fpr, tpr, AUC, accuracy, sensitivity, specificity = get_evaluation_res(y_val, y_pred, y_pred_proba)
            AUC_val_list.append(AUC)
            #print("Current model: %s, current hyper-parameter: %s, current AUC score: %.8f" % (mtd, str(hyperparam_list[idx]), AUC) )
        # Choose the optimal regr
        best_hyperparameter = hyperparam_list[np.argmax(AUC_val_list)]
        print("Best hyper-parameter: %s" % str(best_hyperparameter))
# =============================================================================
#             Training after searching hyper-parameters
# =============================================================================
    x_train = np.concatenate((x_train, x_val), 0)
    y_train = np.concatenate((y_train, y_val), 0)
    if mtd != "lightGBM":
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        y_pred_proba = regr.predict_proba(x_test)[:,1]
        if mtd == "GBM":
            explainer = shap.TreeExplainer(regr)
            shap_values = explainer.shap_values(x_train)            
            shap.summary_plot(shap_values, colnames)
            shap.summary_plot(shap_values, x_train, colnames,plot_type="bar")
            shap_interaction_values = explainer.shap_interaction_values(x_train)
            shap.summary_plot(shap_interaction_values, x_train)
            plots.append(shap.force_plot(explainer.expected_value, shap_values[0,:],colnames,figsize=(15, 3)))
            shap.force_plot(explainer.expected_value, shap_values[0,:],colnames,figsize=(15, 3), matplotlib=True)
            plots.append(shap.force_plot(explainer.expected_value, shap_values,colnames))
    if mtd == "lightGBM":
        params = {}
        evals_result = {} 
        params['learning_rate'] = 0.02
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
        params['num_leaves'] = 11
        params['min_data'] = 1
        params['max_depth'] = best_hyperparameter
        #dealing with imbalance data
        params['pos_bagging_fraction'] = 1
        params['neg_bagging_fraction'] = 0.2
        params['bagging_freq'] = 2
        params['bagging_seed'] = 40
        d_train = lgb.Dataset(x_train, label=y_train)
        clf = lgb.train(params, d_train, num_boost_round = 100, evals_result=evals_result)
        #Prediction
        y_pred_proba = clf.predict(x_test)
        #convert into binary values
        y_pred = (y_pred_proba > .5).astype(int)
    if mtd == "GBM":
        GBM_regr = regr
    fpr, tpr, AUC, accuracy, sensitivity, specificity = get_evaluation_res(y_test, y_pred, y_pred_proba)
    result = {'model': mtd, 'AUC': AUC, 'accuracy': accuracy, 'sensitivity': sensitivity,\
              'specificity': specificity, 'best hyperparameter': best_hyperparameter}
    performance_table = performance_table.append(result, ignore_index = True)
    plt.figure(figsize=(12,6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='%s (AUC = %0.4f%%)' % (mtd, 100*auc(fpr, tpr)))
    plt.axes().set_aspect('equal')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    font1 = {'family' : 'Arial',
    'weight' : 'normal',
    'size'   : 13,
    }
    plt.xlabel('1-Specificity',font1)
    plt.ylabel('Sensitivity',font1)
    plt.title('Receiver Operating Characteristic Curve',font1)    
    plt.legend(loc="lower right",prop=font1)
    plt.savefig("model_" + mtd + "_test_AUC.png",dpi=300)
performance_table.T.to_csv('performance_table.csv')

shap.initjs()
plots[0]
plots[1]