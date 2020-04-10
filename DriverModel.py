import os
import csv
import math
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

def prevent_singular_matrix(X):
    '''inputs: proposed feature matrix for a  regression model
    outputs: selection of redundant columns that will cause 
    proposed feature matrix X to produce an error if fit to y'''
    rank_test = pd.DataFrame(X.copy(),dtype=np.float64)
    for i in range(rank_test.shape[1]):
        i = min(i,rank_test.shape[1])
        if i %50==0:
            print(round(i/rank_test.shape[1],1))
        df_to_rank = rank_test.iloc[:,:i+1]
        if i == np.linalg.matrix_rank(df_to_rank):
            problem_column=df_to_rank.columns[i]
            print(i,np.linalg.matrix_rank(df_to_rank),problem_column)
            rank_test.drop(columns=problem_column,inplace=True)

def make_scatter_matrix(X):
    ''' optional EDA function for scatterplots '''
    scatter_matrix = pd.plotting.scatter_matrix(df,figsize=(75,75),
                              diagonal='kde',alpha=0.2)

    [s.xaxis.label.set_rotation(300) for s in scatter_matrix.reshape(-1)]
    [s.yaxis.label.set_rotation(60) for s in scatter_matrix.reshape(-1)]

    #May need to offset label when rotating to prevent overlap of figure
    [s.get_yaxis().set_label_coords(-0.3,0.5) for s in scatter_matrix.reshape(-1)]

    [plt.setp(item.yaxis.get_label(), 'size', 20) for item in scatter_matrix.ravel()]
    #x labels
    [plt.setp(item.xaxis.get_label(), 'size', 20) for item in scatter_matrix.ravel()]

    #Hide all ticks
    # [s.set_xticks(()) for s in sm.reshape(-1)]
    # [s.set_yticks(()) for s in sm.reshape(-1)]

def make_driver_model(input_data):
    df = input_data.copy()
    df.sort_values(by='Feedback_DateTime',inplace=True)
    df.drop_duplicates(subset='Feedback_TenantId',keep='last',inplace=True)

    all_null_list = []
    for col in df.columns:
        if all(df[col].isnull()):
               all_null_list.append(col)
    var_list = []
    for col in df.columns:
        if df[col].nunique()>1:
               var_list.append(col)

    df = df.loc[:,df.columns.isin(var_list)]

    #Filter out ID columns
    df = df.loc[:,~df.columns.str.contains('ID')]
    df = df.loc[:,~df.columns.str.contains('Id')]
    df = df.loc[:,~df.columns.str.contains('Verbatim')]

    #Filter out redundant Time columns
    time_to_drop = [#'Tenant_CompanyLastDirSyncTime','Tenant_PasswordSyncTime',
        'LastTicket_CreatedTime','LastTicket_ClosedTime'
                   ]
    for x in time_to_drop:
        if x in df.columns:
            df.drop(columns=x,inplace=True)
    df = df.loc[:,~df.columns.str.contains('Date')]

    cats = df.columns[df.dtypes==object]
    #Filter out non-numeric levels with too many levels (define threshold for how many is too much)
    too_many_levels = []
    threshold = 100
    for c in cats:
        if len(df[c].value_counts()) > threshold:
            too_many_levels.append(c)
    df.drop(columns=too_many_levels,inplace=True)

    redundant_region = [#'Tenant_Country','Tenant_CountryCode','Feedback_SystemLocale',
        'Tenant_MSSalesRegionName','Tenant_MSSalesSubRegionName',
        'Tenant_MSSalesSubRegionClusterGroupingName',
        #'Tenant_MSSalesCountryCode','Tenant_MSSalesCountryName',
        'Tenant_MSSalesAreaName','Tenant_SignupRegion']
    df.drop(columns=redundant_region,inplace=True)

    # others_to_drop = ['Tenant_DataCenterInstance',
    #     'Tenant_MSSalesSubsidiaryName']
    # df.drop(columns=others_to_drop,inplace=True)

    df = df.loc[df['Subscription_ChannelNames'].str.contains('DIRECT') | df['Subscription_ChannelNames'].str.contains('VL')|\
                  df['Subscription_ChannelNames'].str.contains('GODADDY.COM, LLC_EE83E790-62B7-4CCB-A1B6-26CAC3C77A8C')|\
                  df['Subscription_ChannelNames'].str.contains('RESELLER')] 


    y = df['NPS'].replace(100,0).replace(-100,1)
    X = df.drop(columns=['NPS','Feedback_Rating','Feedback_RatingValue'])

    StrTo_Cat = df.columns[df.dtypes=='object']
    for x in StrTo_Cat:
       # print(x)
        X.loc[X[x].isnull(), x] ='No_Data'
        X[x]= X[x].astype('category')
        X = pd.concat([X, pd.get_dummies(X[x], prefix=x, prefix_sep='_',)], axis = 1)
        del X[x]
        #pd.data[StrTo_Cat]= data[StrTo_Cat].astype('category')
    X = X.loc[:,~X.columns.str.contains('No_Data')]
        

    X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.2)
    rf = RandomForestClassifier(random_state=0).fit(X_train,y_train)
    rf.score(X_test,y_test)

    learners = rf.feature_importances_.argsort()[::-1]
    features = pd.DataFrame(X.columns[learners], rf.feature_importances_[learners])
    features_original = features[features.index>0.005]
    features_original.columns = ['feature_names']


    rf = RandomForestClassifier(random_state=0)
    param_grid = { 
        'n_estimators': [10,50, 100, 250, 500],
        'max_depth': [2,5,10,25,None]}

    CV_rfc = GridSearchCV(estimator=rf, n_jobs=-1, param_grid=param_grid, verbose=10, scoring='neg_log_loss',cv= 5)
    CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_params_, CV_rfc.best_score_)

    print('Model Improvement After Parameter Tuning: '+str(round(RandomForestClassifier(random_state=0).fit(X_train,
                                               y_train).score(X_test,y_test),2)))


    rfc = RandomForestClassifier(random_state = 0,
                                 **CV_rfc.best_params_,oob_score=True)
    rfc.fit(X_train,y_train)
    print('to: '+str(round(rfc.score(X_test,y_test),2)))

    learners = rfc.feature_importances_.argsort()[::-1]
    features = pd.DataFrame(X.columns[learners], rfc.feature_importances_[learners])
    features_tuned = features[features.index>0.005]
    features_tuned.columns = ['feature_names']

    feature_cols = features_original['feature_names'].to_list()+features_tuned['feature_names'].to_list()
    feature_cols = list(set(feature_cols))

    X_subset = X.loc[:,feature_cols].fillna(0)
#     rank_test = X_subset.copy()
#     linear_dependencies = []
#     for i in range(rank_test.shape[1]):
#         df_to_rank = rank_test.iloc[:,:i+1]
#         if i == np.linalg.matrix_rank(df_to_rank):
#             print(i,np.linalg.matrix_rank(df_to_rank))
#             problem_column = df_to_rank.columns[i]
#             linear_dependencies.append(problem_column)
#     print(linear_dependencies)
#     X_subset.drop(columns=linear_dependencies,inplace=True)

    #redundant columns check
#     for i in range(X_subset.shape[1]):
#         if i < X_subset.shape[1]:
#             if i %50==0:
#                 print(round(i/X_subset.shape[1],1))
#             df_to_rank = X_subset.iloc[:,:i+1]
#             if i == np.linalg.matrix_rank(df_to_rank):
#                 problem_column=df_to_rank.columns[i]
#                 print(i,np.linalg.matrix_rank(df_to_rank),problem_column)
#                 X_subset.drop(columns=problem_column,inplace=True)

    prevent_singular_matrix(X_subset)

    logit = sm.Logit(y, sm.add_constant(X_subset.T.drop_duplicates().T))
    flogit = logit.fit()

    print(flogit.summary())

    coefficients = flogit.summary2().tables[1]
    coefficients = coefficients[coefficients['P>|z|']<0.1]
    coefficients['FinalProbability'] = np.exp(coefficients['Coef.'].round(1))*y.value_counts(normalize=True).loc[1] #- 0.5
    coefficients['FinalProbability'] = coefficients['FinalProbability'].mask(coefficients['FinalProbability']>=1,0.99)
    coefficients['Lift'] = coefficients['FinalProbability'] - y.value_counts(normalize=True).loc[1]
    return flogit, coefficients

df = sys.argv[1] #point to the data file you are using in terminal
model, scorecard = make_driver_model(df)
scorecard.to_csv('DriverModelScorecard.csv')


