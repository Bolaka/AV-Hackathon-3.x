# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:31:52 2015

@author: bolaka

This solution measures 0.84020 AUC on private leaderboard. (alas I did not submit it!)

"""

from __future__ import division
from collections import defaultdict
from glob import glob
import sys

# suppress pandas warnings
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)

# imports
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
import pandas as pd
from ggplot import *
from mlclassificationlibs import * 
from numpy.random import seed
from scipy.special import cbrt
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

# reproduce results
seed(786)

# load the training and test sets
dateparse = lambda x: pd.datetime.strptime(x, '%d-%b-%y')
data = pd.read_csv('Train.csv', index_col = 'ID', encoding = 'latin1', parse_dates = ['Lead_Creation_Date'], date_parser=dateparse) # ,'DOB'
test=pd.read_csv('Test.csv', index_col = 'ID', parse_dates = ['Lead_Creation_Date'], encoding = 'latin1', date_parser=dateparse) # ,'DOB'

# sum up Missing
def sumMissing(s):
    return s.isnull().sum()

# fix the format of DOB & convert both Lead_Creation_Date & DOB to int
def fixDates(data):
    data['DOB_yr'] = [item.split('-')[2] for item in data['DOB']]
    data.DOB_yr = '19' + data.DOB_yr
    data['DOB_mon'] = [item.split('-')[1] for item in data['DOB']]
    data['DOB_day'] = [item.split('-')[0] for item in data['DOB']]
    data.DOB = data.apply(lambda x: pd.datetime.strptime("{0} {1} {2} 00:00:00".format(x['DOB_yr'],x['DOB_mon'], x['DOB_day']), "%Y %b %d %H:%M:%S"),axis=1)

    # drop extra features
    data.drop( [ 'DOB_mon', 'DOB_day' ] , axis=1, inplace=True) # 'DOB_yr', 
    data.DOB_yr = [int(x) for x in data.DOB_yr]

    # convert dates to ordinal
    data['Lead_Creation_Date'] = data['Lead_Creation_Date'].apply(lambda x: x.toordinal())    
    data['DOB'] = data['DOB'].apply(lambda x: x.toordinal())  
    
    return data

def dictMap(listOfMajors, non_major):
    mapped_dict = {}
    for i, major in enumerate(reversed(listOfMajors)):
        mapped_dict[major] = (i+1)
    mapped_dict[non_major] = 0
    return mapped_dict

def dictMap0(listOfMajors):
    mapped_dict = {}
    for i, major in enumerate(reversed(listOfMajors)):
        mapped_dict[major] = i
    return mapped_dict

# Pre-process data
def dataPreprocessing(data, test):
    # add the target columns to test data as 9999
    test['LoggedIn'] = 9999
    test['Disbursed'] = 9999
    
    # combine the training and test datasets for data preprocessing
    combined = pd.concat( [ data, test ] )    
    
    combined = fixDates(combined)
    
    # Gender - Female = 0, Male = 1
    combined['Gender'] = combined['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
    
    # Filled_Form - N = 0, Y = 1
    combined['Filled_Form'] = combined['Filled_Form'].map( {'N': 0, 'Y': 1} ).astype(int)
    
    # Device_Type - Mobile = 0, Web-browser = 1
    combined['Device_Type'] = combined['Device_Type'].map( {'Mobile': 0, 'Web-browser': 1} ).astype(int)
    
    # Mobile_Verified - N = 0, Y = 1
    combined['Mobile_Verified'] = combined['Mobile_Verified'].map( {'N': 0, 'Y': 1} ).astype(int)
    
    # City
    city_counts = data.City.value_counts()
    major_cites = city_counts.index[:11]
    combined.loc[ ~combined['City'].isin(major_cites), 'City' ] = 'Non-major city'
    mapped_cities = dictMap(major_cites, 'Non-major city')
    combined['City'] = combined['City'].map( mapped_cities ).astype(int)
    
#     Employer_Name
    # clean up outliers in TCS - the most occuring and most 'trusted' employer
    data.loc[ data.Employer_Name.isin( ['TATA CONSALTANCY SERVICES', 'TATA CONSULTANCY SERVICE', 'TATA CONSULTANCY SERVICES', 
                                        'TATA CONSULTANCY SERVICES LIMITED', 'TATA CONSULTANCY SERVICES LTD (TCS)CONSUL'] ) , 'Employer_Name' ] = 'TATA CONSULTANCY SERVICES LTD (TCS)'
    combined.loc[ combined.Employer_Name.isin( ['TATA CONSALTANCY SERVICES', 'TATA CONSULTANCY SERVICE', 'TATA CONSULTANCY SERVICES', 
                                        'TATA CONSULTANCY SERVICES LIMITED', 'TATA CONSULTANCY SERVICES LTD (TCS)CONSUL'] ) , 'Employer_Name' ] = 'TATA CONSULTANCY SERVICES LTD (TCS)'
    # TODO - clean similar outliers in other popular employers if any
    
    # ranking employers by Disbursed sum
    employer_groups = data.groupby('Employer_Name')['Disbursed'].sum()
    major_employers = list(employer_groups.order()[-20:].index)
    major_employers.remove('0')
    major_employers.remove('TYPE SLOWLY FOR AUTO FILL')
    combined.loc[ ~combined['Employer_Name'].isin(major_employers), 'Employer_Name' ] = 'Non-major employer'
    
    mapped_employers = dictMap(major_employers, 'Non-major employer')
    combined['Employer_Name'] = combined['Employer_Name'].map( mapped_employers ).astype(int)
    
    # Salary_Account
    bank_counts = data.Salary_Account.value_counts()
    major_banks = list(bank_counts.index[:20])
    combined.loc[ ~combined['Salary_Account'].isin(major_banks), 'Salary_Account' ] = 'Non-major bank'
    mapped_banks = dictMap(major_banks, 'Non-major bank')
    combined['Salary_Account'] = combined['Salary_Account'].map( mapped_banks ).astype(int)
    
    # Var1
    var1_counts = data.Var1.value_counts()
    major_var1 = list(var1_counts.index[:7]) # 
    combined.loc[ ~combined['Var1'].isin(major_var1), 'Var1' ] = 'Non-major var1'
    mapped_var1 = dictMap(major_var1, 'Non-major var1')
    combined['Var1'] = combined['Var1'].map( mapped_var1 ).astype(int)
    
    # Var2
    var2_counts = data.Var2.value_counts()
    major_var2 = list(var2_counts.index)
    mapped_var2 = dictMap0(major_var2)
    combined['Var2'] = combined['Var2'].map( mapped_var2 ).astype(int)
    
    # Source
    source_counts = data.Source.value_counts()
    major_source = list(source_counts.index[:7])
    combined.loc[ ~combined['Source'].isin(major_source), 'Source' ] = 'Non-major source'
    mapped_source = dictMap(major_source, 'Non-major source')
    combined['Source'] = combined['Source'].map( mapped_source ).astype(int)

    # Transformations to correct skewness...
    combined.Monthly_Income = combined.Monthly_Income.apply(np.sqrt) 
    
    combined.Loan_Amount_Applied = combined.Loan_Amount_Applied.apply(np.sqrt)
    
    combined.Existing_EMI = [np.power(x, (float(1)/3)) for x in combined.Existing_EMI ] # combined.Existing_EMI.apply(np.sqrt)
    
    combined.Loan_Amount_Submitted = combined.Loan_Amount_Submitted.apply(np.sqrt)
    
    combined.DOB_yr = [np.log(x + 1) for x in combined.DOB_yr ] 
    
    combined.Processing_Fee = combined.Processing_Fee.apply(np.sqrt)
    
    # removing outliers    
    combined = removeOutliers(combined)
    
#    # sum up missing    
#    combined['missingness'] = combined.apply(sumMissing, 1)     
    
#    # fill missing
#    combined = fillMissingby9999(combined)
    
    # separate again into training and test sets
    data = combined.loc[ combined.Disbursed != 9999 ]
    test = combined.loc[ combined.Disbursed == 9999 ]
    
    # remove the target columns from test data
    test.drop(['LoggedIn','Disbursed'], axis=1, inplace=True)    
    
    return data, test

# tolerance 10 standard deviations - replace by NaN
def removeOutliers(data):
    # remove outliers (replacing by null for std > outlier_cutoff)
    outlier_cutoff = 10
    for feature in data.columns:
        if feature in ['LoggedIn', 'Disbursed', 'Employer_Name']:
            continue
        
        data[feature + '_std'] = np.abs( (data[feature] - data[feature].mean()) / data[feature].std() )
        if len( data.loc[ data[ feature + '_std' ] > outlier_cutoff, feature ] ) > 0:
            print('removing outliers in ', feature, ':\n', data.loc[ data[ feature + '_std' ] > outlier_cutoff, feature ])
            data.loc[ data[feature + '_std'] > outlier_cutoff, feature ] = float('nan')
        data.drop( [feature + '_std'], axis=1, inplace=True)
    return data

# fill missing values by -9999
def fillMissingby9999(data):
    data.fillna(-9999,inplace=True)
    return data

# Cross validation & modeling
def modeling(data, test, classifier):
    print()
    print('class % before split = ', data.Disbursed.sum(), '/', len(data.Disbursed))
    
    # Divide data roughly into train and unseen Validate
    data['is_train'] = np.random.uniform(0, 1, len(data)) <= .80
    train, validate = data[data['is_train']==True], data[data['is_train']==False]
#    print('unseen validation set has', len(validate), 'records')
    
    # the feature set
    features=[
    #'DOB',
    'DOB_yr', 
#    'Lead_Creation_Date',
    'Gender',
    'City',
    'Monthly_Income',
    'Loan_Amount_Applied',
    'Loan_Tenure_Applied',
    'Existing_EMI',
    'Employer_Name',
    'Salary_Account',
    'Mobile_Verified',
    'Var5',
    'Var1',
    'Loan_Amount_Submitted',
    'Loan_Tenure_Submitted',
    'Interest_Rate',
    'Processing_Fee',
    'EMI_Loan_Submitted',
    'Filled_Form',
    'Device_Type',
    'Var2',
    'Source',
    'Var4',
#    'missingness'
    ]
    
    # X and Y
    x = data[list(features)].values
    y = data['Disbursed'].values
    x_train = train[list(features)].values
    x_validate = validate[list(features)].values
    y_train = train['Disbursed'].values
    y_validate = validate['Disbursed'].values
    x_test = test[list(features)].values
    
#    print('class % in train = ', sum(y_train), '/', len(y_train))
#    print('class % in validation = ', sum(y_validate), '/', len(y_validate))
    
    print()
    
    # Stratified 5-fold cross validation
    number_of_folds = 5
    skf = StratifiedKFold(y_train, n_folds=number_of_folds, shuffle=False)
    y_pred = np.array(y_train.copy()).astype('float')
    
    # Iterate through folds
    importance_cv = pd.Series(data=None, index=features)
    fold = 0
    cv_auc = []
    for train_index, test_index in skf:
        x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        
        ensemble = classifier.fit(x_train_cv, y_train_cv)
    
        #Plot ROC_AUC curve and cross validate
        disbursed = ensemble.predict_proba(x_test_cv)
        disbursed1 = ensemble.predict_proba(x_train_cv)
        
        y_pred[test_index] = disbursed[:,1]
        fpr, tpr, _ = roc_curve(y_test_cv, disbursed[:,1])
        roc_auc = auc(fpr, tpr)
        fpr1, tpr1, _ = roc_curve(y_train_cv, disbursed1[:,1])
        roc_auc1 = auc(fpr1, tpr1)
        print('fold', str(fold+1),'--- train ROC_AUC = ', roc_auc1, '--- test ROC_AUC = ', roc_auc)    
        cv_auc.append(roc_auc)
        
        # fold feature importance
        mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
        ts = pd.Series(ensemble.booster().get_fscore())
        ts.index = ts.reset_index()['index'].map(mapFeat)
        importance_cv = importance_cv.add(ts, fill_value=0)
        fold += 1
    
    # plot the feature importance accross folds
    importance_cv.dropna().order().plot(kind="barh", title=("features importance accross 10-folds")) # [-15:]
    
    print('mean AUC', np.average(cv_auc), 'STD AUC = ', np.std(cv_auc))
    
    #Plot ROC_AUC curve and cross validate
    #disbursed = ensemble.predict_proba(x_validate)
    fpr_cv, tpr_cv, _ = roc_curve(y_train, y_pred)
    roc_auc_cv = auc(fpr_cv, tpr_cv)
    print('Cross validation ROC_AUC = ', roc_auc_cv)
    
    # train the model on training data and test on validation set
    train_ensemble = classifier.fit(x_train, y_train)
    # TODO try this out later
    #train_ensemble = xgb.XGBClassifier(missing=float('nan')).fit(x_train, y_train, eval_set = [(x_validate, y_validate)], eval_metric="auc", early_stopping_rounds=10)
    train_disbursed = train_ensemble.predict_proba(x_validate)
    fpr_val, tpr_val, _ = roc_curve(y_validate, train_disbursed[:,1])
    roc_auc_val = auc(fpr_val, tpr_val)
    print('Unseen validation ROC_AUC = ', roc_auc_val)

    # train the model on entire data and use it to predict the test set
    whole_ensemble = classifier.fit(x, y)
    
    #Predict for test data set and export test data set 
    test_disbursed = whole_ensemble.predict_proba(x_test)
    
    return test_disbursed[:,1]

# approach 1
# TODO - parameter tuning
model1 = xgb.XGBClassifier(max_depth=3, n_estimators=700, learning_rate=0.05)
data1, test1 = dataPreprocessing(data.copy(deep=True), test.copy(deep=True))
solution_best1 = modeling(data1, test1, model1) # 

test1['Disbursed'] = solution_best1
test1.to_csv('Solution_xgb.csv', columns=['Disbursed'],index=True)

# TODO ensembling
## approach 2
#model2 = RandomForestClassifier(n_estimators=700)
#data2, test2 = dataPreprocessing2(data.copy(deep=True), test.copy(deep=True))
#solution_best2 = modeling(data2, test2, model2)
#
#test2['Disbursed'] = solution_best2
#test2.to_csv('Solution2.csv', columns=['Disbursed'],index=True)

## approach 3
#model3 = GradientBoostingClassifier(n_estimators=700)
#data3, test3 = dataPreprocessing1(data.copy(deep=True), test.copy(deep=True))
#solution_best3 = modeling(data3, test3, model3)
#
#test3['Disbursed'] = solution_best3
#test3.to_csv('Solution_gbm.csv', columns=['Disbursed'],index=True)
#
## check the correlation between the 2 approaches
#plt.scatter(solution_best1,solution_best3)
#plt.show()
#print('Correlation between the 2 approaches = ', np.corrcoef(solution_best1,solution_best2)[0][1])
#
#glob_files = "Solution*.csv"
#loc_outfile = "rankedavg.csv"
#
#def rankavg_ensemble(glob_files, loc_outfile):
#    with open(loc_outfile,"w") as outfile:
#        all_ranks = defaultdict(list)
#        for i, glob_file in enumerate( glob(glob_files) ):
#            file_ranks = []
#            print("parsing:", glob_file)
#            for e, line in enumerate( open(glob_file) ):
#                if e == 0 and i == 0:
#                    outfile.write( line )
#                elif e > 0:
#                    r = line.strip().split(",")
#                    file_ranks.append( (float(r[1]), e, r[0]) )
#            for rank, item in enumerate( sorted(file_ranks) ):
#                all_ranks[(item[1],item[2])].append(rank)
#        average_ranks = []
#        for k in sorted(all_ranks):
#            average_ranks.append((sum(all_ranks[k])/len(all_ranks[k]),k))
#        ranked_ranks = []
#        for rank, k in enumerate(sorted(average_ranks)):
#            ranked_ranks.append((k[1][0],k[1][1],rank/(len(average_ranks)-1)))
#        for k in sorted(ranked_ranks):
#            outfile.write("%s,%s\n"%(k[1],k[2]))
#        print("wrote to %s"%loc_outfile)
#
#rankavg_ensemble(glob_files, loc_outfile)