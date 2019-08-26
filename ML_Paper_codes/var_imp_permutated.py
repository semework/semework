#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:33:43 2018
sourced from: https://github.com/parrt/random-forest-importances/blob/master/src/rfpimp.py
@author: mulugetasemework
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import sys 
from pathlib import Path
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

sys.path.append("/Users/mulugetasemework/Dropbox/Phyton")

from  Models_and_methods import models_and_methods
import time
import math
start = time.time()

# %% clean and reset iphython console if needed
#    %clear
#    %reset

#%% pareameters and direcotries
path = "/Users/mulugetasemework/Dropbox/Phyton"

data_path = '/Users/mulugetasemework/Dropbox/R/OtherCSVs/1MetaTable1_filtered3.csv'
results_path = '/Users/mulugetasemework/Documents/Python/Results/'

#%% important parameters to play with
sampling_percentage = 1 # how much of the data we would like to use,in fractions, upto "1", meaning use all
predict_binary = 0
drop_columns_with_specific_phrase = 1

#%%  create a y-label based on what is predicted
if predict_binary  == 1:
    ylabel_text = "binarized_outcome"
else:
    ylabel_text = "continuous_outcome"

# add sample size to file name for ease of recall
if sampling_percentage < 1:
    ylabel_text = (ylabel_text +"_" + str(Path(data_path).stem) + str(int(math.ceil(sampling_percentage*100))) + "_percent_data_used") 
else:
    ylabel_text = (ylabel_text +"_" + str(Path(data_path).stem)  + "_All_data_used") 

#%%
save_csv = 1
save_fig = 1

#%% these parameters are unique for this code only, i.e they aren't needed for prediction codes

do_over_all_var_importance = 1
plot_importance_ranks = 1
plt.rcdefaults()
plt.close("all")
imputed_already = 0
#%% import data
TDData = pd.read_csv(data_path,low_memory=False, error_bad_lines=False)
#safe keeping forpossible reuse, without imoorting CSV
TDData_orig = TDData
#drop the pesky "Unnamed: 0" column, if exists
TDData.drop([col for col in TDData.columns if "Unnamed" in col], axis=1, inplace=True)

#%% clean up data, for instance, drop columns with names that have "memory"

#drop eye pos and other irrelevant rows
TDData = TDData[(TDData.baseline_pre_vs_eyepos_max>0.05) & (TDData.baseline_pre_vs_eyepos_median>0.05) & (TDData.baseline_pre_vs_eyepos_mean>0.05)]

drop_these_columns = ['baseline_pre_vs_eyepos_max','baseline_pre_vs_eyepos_median',
                 'TaskType','visual_vs_baseline_ttest_h',
                 'baseline_pre_vs_eyepos_mean', 'new_saccade_baseline_max_ranksum']
for df3 in drop_these_columns:
    if df3 in TDData.columns:
        del TDData[df3]

# keep column nams for later use
colNames =  TDData.columns

to_be_predicted = TDData.memory_response_strength
# drop target variables (i.e. columns with "memory" in their names)
memCols =   TDData.filter(regex='mem').columns
#binarize
if predict_binary ==1:
    print("binarizing target")
    to_be_predicted = (to_be_predicted<0.05)*1

if drop_columns_with_specific_phrase == 1:
#drop all "memory" columns and put data to predicted in last column
    TDData.drop([col for col in TDData.columns if "mem" in col], axis=1, inplace=True)
    TDData.drop([col for col in TDData.columns if "indic" in col], axis=1, inplace=True)
    TDData.drop([col for col in TDData.columns if "latency" in col], axis=1, inplace=True)
    
# remove to_be_predicted from its original location, and put as last col
TDData = TDData.loc[:, TDData.columns != 'memory_response_strength']
    
#move target variable to the last columns
TDData = pd.DataFrame(pd.concat([TDData,pd.DataFrame(to_be_predicted)],axis=1))
 
  #%%  cleanup: NANs and zero standard columns
TDData = TDData.dropna(axis=1, how='all')
TDData = TDData.dropna(axis=0, how='all')
TDData = TDData.loc[:, (TDData != TDData.iloc[0]).any()]
TDData = TDData.reset_index(drop=True)

#%% save data for later use
TDData.to_csv(os.path.join(path,results_path,str('Raw_'+ylabel_text+'.csv')), float_format='%.5f')

# %% save only non-mem and cleaned_up predictor columns from the above raw file
feaure_labels = os.path.join(path,results_path,str('Predictive_features.csv'))
 
pred_features_data = pd.read_csv(feaure_labels,low_memory=False, error_bad_lines=False)
clean_data = TDData.copy()
clean_data = clean_data.reindex(columns=pred_features_data.Feature)
clean_data.to_csv(os.path.join(path,results_path,str('Clean_raw_'+ylabel_text+'.csv')), float_format='%.5f')

#%% copy claen data back to TDData
TDData = clean_data.copy()
      
#%% data
X = TDData.iloc[:,:TDData.shape[1]-1]

print(np.unique(X))
if predict_binary ==1:
    y = (TDData.iloc[:,-1]<0.05)*1
else:
    y = (TDData.iloc[:,-1]<0.05) 

#add random data for sanity check
X = X.assign(R_A_N_D_O_M=np.random.random(size=len(y)))

X_train, X_test, y_train, y_test  =  \
        train_test_split(X, y,  test_size = 0.4, random_state=12345)

"""scale  data"""
y_test_orig = y_test.copy()
#save column names for later use
colNames =  X_train.columns
print(np.unique(X_train))
X_train, X_test, y_train, y_test, scaler_y = models_and_methods.scale_data(X_train, X_test, y_train, y_test)
print(np.unique(X_train))
# put column names back
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

X_train.columns = colNames

X_test.columns = colNames

#%%   do overall var imp , ie. all days as one

if do_over_all_var_importance == 1:
   imp,imp_pos = models_and_methods.over_all_var_importance(predict_binary,ylabel_text,save_fig,results_path,X_train,y_train,plot_importance_ranks)
   imp_cols_data = TDData.loc[ : , imp_pos.index] #.reindex(columns=imp_pos.index)
   imp_data = pd.DataFrame(pd.concat([imp_cols_data,pd.DataFrame(y_train)],axis=1))
   imp_data.columns.values[-1] = 'memory_response_strength'
   target_continuous = pd.DataFrame(TDData.iloc[:,-1])
   target_continuous.rename(columns={'0': 'memory_response_strength'}, inplace=True)

   print("\n===================  var importances calculated   =========")
    #%% save data
   if save_csv == 1:
    imp.to_csv(os.path.join(path,results_path,str('overall_varImp_MSE_'+ylabel_text+'.csv')),index=False)
    imp_pos.to_csv(os.path.join(path,results_path,str('positive_overall_varImp_MSE_'+ylabel_text+'.csv')),index=True)
    imp_data.to_csv(os.path.join(path,results_path,str('imp_data_'+ylabel_text+'.csv')),index=False)
    target_continuous.to_csv(os.path.join(path,results_path,str('target_continuous_'+ylabel_text+'.csv')),index=False)
    

    print("\n===================  var importances saved   =========")
