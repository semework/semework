#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:45:52 2019

@author: mulugetasemework
"""

import numpy as np
from numpy import newaxis
#from utils import Timer
#from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
#from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble.forest import _generate_unsampled_indices
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tempfile
from os import getpid
import tensorflow as tf
import warnings
import os

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#%%
"""scale data"""

class models_and_methods():
    """A class for an methods and building models"""
    
    def scale_data(X_train,X_test,y_train,y_test):
        
        #scaler for matrices
        scaler_mat = MinMaxScaler(copy=True, feature_range=(0, 1))
        #scaler for test/train array, model will be returned for inverse transform
        scaler_y = MinMaxScaler(copy=True, feature_range=(0, 1))
        
        data_train =  pd.DataFrame(X_train.reset_index(drop=True))
        data_test =   pd.DataFrame(X_test.reset_index(drop=True))
        #fit and transform train and test matrices
        scaler_mat.fit(data_train)
        data_train = scaler_mat.transform(data_train)
        scaler_mat.fit(data_test)
        data_test= scaler_mat.transform(data_test)
        
        
        if isinstance(y_train, (np.ndarray, np.generic) ):
            scaler_y.fit(pd.DataFrame(y_train))
            y_train_scaled = scaler_y.transform(pd.DataFrame(y_train))
        else:
            scaler_y.fit(pd.DataFrame(y_train.values))
            y_train_scaled = scaler_y.transform(pd.DataFrame(y_train.values))
        
        #the following model will be returned for later use (to inverse transform prediction and test data)
        
        if isinstance(y_test, (np.ndarray, np.generic) ):
             scaler_y.fit(pd.DataFrame(y_test))
             y_test_scaled = scaler_y.transform(pd.DataFrame(y_test))
        else:
             scaler_y.fit(pd.DataFrame(y_test.values))
             y_test_scaled = scaler_y.transform(pd.DataFrame(y_test.values))
         
        return data_train, data_test, y_train_scaled.flatten(), y_test_scaled.flatten(), scaler_y, 

    def stemplot_importances(df_importances,
                             yrot=0,
                             label_fontsize=10,
                             width=6,
                             minheight=2.5,
                             vscale=1.0,
                                  imp_range1=-.002,
                             imp_range2=.15,
                             color='#375FA5',
                             bgcolor=None,  # seaborn uses '#F1F8FE'
                             xtick_precision=2,
                             title=None):
        GREY = '#444443'
        I = df_importances
        unit = 1
    
        imp = I.Importance.values
        mindrop = np.min(imp)
        maxdrop = np.max(imp)
        imp_padding = 0.0002
        imp_range=(imp_range1,imp_range2)
        imp_range = (min(imp_range[0], mindrop - imp_padding), max(imp_range[1], maxdrop))
    
        barcounts = np.array([f.count('\n')+1 for f in I.index])
        N = np.sum(barcounts)
        ymax = N * unit
        # print(f"barcounts {barcounts}, N={N}, ymax={ymax}")
        height = max(minheight, ymax * .27 * vscale)
    
        plt.close()
        fig, ax = plt.subplots(1, 2,figsize=(10, 8))
    #    fig = plt.figure(figsize=(width,height))
        fig = plt.figure(figsize=(10,8))
        ax = plt.gca()
        ax.set_xlim(*imp_range)
        ax.set_ylim(0,ymax)
        ax.spines['top'].set_linewidth(.3)
        ax.spines['right'].set_linewidth(.3)
        ax.spines['left'].set_linewidth(.3)
        ax.spines['bottom'].set_linewidth(.3)
        if bgcolor:
            ax.set_facecolor(bgcolor)
    
        yloc = []
        y = barcounts[0]*unit / 2
        yloc.append(y)
        for i in range(1,len(barcounts)):
            wprev = barcounts[i-1]
            w = barcounts[i]
            y += (wprev + w)/2 * unit
            yloc.append(y)
        yloc = np.array(yloc)
        ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{xtick_precision}f'))
        ax.set_xticks([maxdrop, imp_range[1]])
        ax.tick_params(labelsize=label_fontsize, labelcolor=GREY)
        ax.invert_yaxis()  # labels read top-to-bottom
        if title:
            ax.set_title(title, fontsize=label_fontsize+1, fontname="Arial", color=GREY)
    
        plt.hlines(y=yloc, xmin=imp_range[0], xmax=imp, lw=barcounts*1.2, color=color)
        for i in range(len(I.index)):
            plt.plot(imp[i], yloc[i], "o", color=color, markersize=barcounts[i]+2)
        ax.set_yticks(yloc)
        ax.set_yticklabels(I.index, fontdict={'verticalalignment': 'center'})
        plt.tick_params(
            pad=0,
            axis='y',
            which='both',
            left=False)
    
        # rotate y-ticks
        if yrot is not None:
            plt.yticks(rotation=yrot)
    
        plt.tight_layout()
    
        return PimpViz()
    
    def plot_importances2(df_importances,
                         yrot=0,
                         label_fontsize=10,
                         width=6,
                         minheight=2.5,
                         vscale=1,
                                 imp_range1=-.002,
                             imp_range2=.15,
                         color='#D9E6F5',
                         bgcolor=None,  # seaborn uses '#F1F8FE'
                         xtick_precision=2,
                         title=None):
        """
        Given an array or data frame of importances, plot a horizontal bar chart
        showing the importance values.
    
        :param df_importances: A data frame with Feature, Importance columns
        :type df_importances: pd.DataFrame
        :param width: Figure width in default units (inches I think). Height determined
                      by number of features.
        :type width: int
        :param minheight: Minimum plot height in default matplotlib units (inches?)
        :type minheight: float
        :param vscale: Scale vertical plot (default .25) to make it taller
        :type vscale: float
        :param label_fontsize: Font size for feature names and importance values
        :type label_fontsize: int
        :param yrot: Degrees to rotate feature (Y axis) labels
        :type yrot: int
        :param label_fontsize:  The font size for the column names and x ticks
        :type label_fontsize:  int
        :param scalefig: Scale width and height of image (widthscale,heightscale)
        :type scalefig: 2-tuple of floats
        :param xtick_precision: How many digits after decimal for importance values.
        :type xtick_precision: int
        :param xtick_precision: Title of plot; set to None to avoid.
        :type xtick_precision: string
        :return: None
    
        SAMPLE CODE
    
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
        X_train, y_train = ..., ...
        rf.fit(X_train, y_train)
        imp = importances(rf, X_test, y_test)
        viz = plot_importances(imp)
        viz.save('file.svg')
        viz.save('file.pdf')
        viz.view() # or just viz in notebook
        """
        GREY = '#444443'
        I = df_importances
        unit = 1
        ypadding = .1
    
        imp = I.Importance.values
        mindrop = np.min(imp)
        maxdrop = np.max(imp)
        imp_padding = 0.0002
        imp_range=(imp_range1,imp_range2)
        imp_range = (min(imp_range[0], mindrop - imp_padding), max(imp_range[1], maxdrop + imp_padding))
    
        barcounts = np.array([f.count('\n')+1 for f in I.index])
        N = np.sum(barcounts)
        ymax = N * unit + len(I.index) * ypadding + ypadding
        # print(f"barcounts {barcounts}, N={N}, ymax={ymax}")
        height = max(minheight, ymax * .2 * vscale)
    
        plt.close()
    #    fig = plt.figure(figsize=(width,height))
        fig = plt.figure(figsize=(10,8))
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42 
        ax = plt.gca()
        ax.set_xlim(*imp_range)
        ax.set_ylim(0,ymax)
        ax.spines['top'].set_linewidth(.3)
        ax.spines['right'].set_linewidth(.3)
        ax.spines['left'].set_linewidth(.3)
        ax.spines['bottom'].set_linewidth(.3)
        if bgcolor:
            ax.set_facecolor(bgcolor)
    
        yloc = []
        y = barcounts[0]*unit / 2 + ypadding
        yloc.append(y)
        for i in range(1,len(barcounts)):
            wprev = barcounts[i-1]
            w = barcounts[i]
            y += (wprev + w)/2 * unit + ypadding
            yloc.append(y)
        yloc = np.array(yloc)
        ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{xtick_precision}f'))
        # too close to show both max and right edge?
        if maxdrop/imp_range[1] > 0.9 or maxdrop < 0.02:
            ax.set_xticks([0, imp_range[1]])
        else:
            ax.set_xticks([0, maxdrop, imp_range[1]])
        ax.tick_params(labelsize=label_fontsize, labelcolor=GREY)
        ax.invert_yaxis()  # labels read top-to-bottom
        if title:
            ax.set_title(title, fontsize=label_fontsize+1, fontname="Arial", color=GREY)
    
        barcontainer = plt.barh(y=yloc, width=imp,
                                height=barcounts*unit,
                                tick_label=I.index,
                                color=color, align='center')
    
        # Alter appearance of each bar
        for rect in barcontainer.patches:
                rect.set_linewidth(.5)
                rect.set_edgecolor(GREY)
    
        # rotate y-ticks
        if yrot is not None:
            plt.yticks(rotation=yrot)
    
        plt.tight_layout()
        plt.pause(0.1)
        return PimpViz()
   
    def oob_regression_r2_score(rf, X_train, y_train):
        """
        Compute out-of-bag (OOB) R^2 for a scikit-learn random forest
        regressor. We learned the guts of scikit's RF from the BSD licensed
        code:
        https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
        """
        X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y = y_train.values if isinstance(y_train, pd.Series) else y_train
    
        n_samples = len(X)
        predictions = np.zeros(n_samples)
        n_predictions = np.zeros(n_samples)
        for tree in rf.estimators_:
            unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
            tree_preds = tree.predict(X[unsampled_indices, :])
            predictions[unsampled_indices] += tree_preds
            n_predictions[unsampled_indices] += 1
    
        if (n_predictions == 0).any():
            warnings.warn("Too few trees; some variables do not have OOB scores.")
            n_predictions[n_predictions == 0] = 1
    
        predictions /= n_predictions
    
        oob_score = r2_score(y, predictions)
        return oob_score
    
    def oob_classifier_accuracy(rf, X_train, y_train):
        """
        Compute out-of-bag (OOB) accuracy for a scikit-learn random forest
        classifier. We learned the guts of scikit's RF from the BSD licensed
        code:
    
        https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L425
        """
        X = X_train.values
        y = y_train.values
    
        n_samples = len(X)
        n_classes = len(np.unique(y))
        predictions = np.zeros((n_samples, n_classes))
        for tree in rf.estimators_:
            unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
            tree_preds = tree.predict_proba(X[unsampled_indices, :])
            predictions[unsampled_indices] += tree_preds
    
        predicted_class_indexes = np.argmax(predictions, axis=1)
        predicted_classes = [rf.classes_[i] for i in predicted_class_indexes]
    
        oob_score = np.mean(y == predicted_classes)
        return oob_score
    
    def sample(X_valid, y_valid, n_samples):
        if n_samples < 0: n_samples = len(X_valid)
        n_samples = min(n_samples, len(X_valid))
        if n_samples < len(X_valid):
            ix = np.random.choice(len(X_valid), n_samples)
            X_valid = X_valid.iloc[ix].copy(deep=False)  # shallow copy
            y_valid = y_valid.iloc[ix].copy(deep=False)
        return X_valid, y_valid
    
    
    def permutation_importances2(rf, X_train, y_train, metric, n_samples=-1):
    
        imp = models_and_methods.permutation_importances_raw(rf, X_train, y_train, metric, n_samples)
        I = pd.DataFrame(data={'Feature':X_train.columns, 'Importance':imp})
        I = I.set_index('Feature')
        I = I.sort_values('Importance', ascending=False)
        return I
    
    def permutation_importances_raw(rf, X_train, y_train, metric, n_samples=-1):
        """
        Return array of importances from pre-fit rf; metric is function
        that measures accuracy or R^2 or similar. This function
        works for regressors and classifiers.
        """
    
        X_sample, y_sample = models_and_methods.sample(X_train, y_train, n_samples)
    
        if not hasattr(rf, 'estimators_'):
            rf.fit(X_sample, y_sample)
    
        baseline = metric(rf, X_sample, y_sample)
        X_train = X_sample.copy(deep=False) # shallow copy
        y_train = y_sample
        imp = []
        for col in X_train.columns:
            save = X_train[col].copy()
            X_train[col] = np.random.permutation(X_train[col])
            m = metric(rf, X_train, y_train)
            X_train[col] = save
            drop_in_metric = baseline - m
            imp.append(drop_in_metric)
        return np.array(imp)
    
        #%% inc node purity function
    
    def do_RF_var_imp(this_category_data_X, this_category_data_y, this_category_data_X_test, this_category_data_y_test):
    #    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
        model = RandomForestClassifier(max_depth=10)
        model.fit(this_category_data_X.astype(int) , this_category_data_y.astype(int)  )
        ## And score it on your testing data.
        model.score(this_category_data_X_test.astype(int), this_category_data_y_test.astype(int))
    
        feature_importances = pd.DataFrame(model.feature_importances_,
           index = this_category_data_X.columns,columns=['importance']).sort_values('importance', ascending=False)
    
        y_pos = np.arange(len(feature_importances))
        performance = feature_importances.importance
        objects = list(feature_importances.index)
    
        y_pred = model.predict(this_category_data_X_test)
        prec = float(np.sum(y_pred == this_category_data_y_test)) / len(this_category_data_y_test)
    
        return performance, y_pos, objects, prec
    
    def over_all_var_importance(predict_this,ylabel_text,save_fig,results_path,X_train,y_train,plot_importance_ranks):
        if predict_this == 1:
            rf = RandomForestClassifier()
        else:    
            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
       
        o_X = X_train.copy().astype(np.int64)
        o_y = y_train.copy().astype(np.int64)
#        o_X['R_A_N_D_O_M'] = np.random.random(size=len(o_y))
#        o_X = o_X.assign(R_A_N_D_O_M=np.random.random(size=len(o_y)))
#
        rf.fit(o_X,o_y)
        
        if isinstance(rf, RandomForestClassifier):
           imp =  models_and_methods.permutation_importances2(rf, pd.DataFrame(o_X), pd.DataFrame(o_y),
                                      models_and_methods.oob_classifier_accuracy)
        elif isinstance(rf, RandomForestRegressor):
           imp =  models_and_methods.permutation_importances2(rf,pd.DataFrame(o_X), pd.DataFrame(o_y),
                                       models_and_methods.oob_regression_r2_score)
        
        
           
           
        if plot_importance_ranks==1:
            models_and_methods.plot_importances2(imp, yrot=0,
                                 label_fontsize=6,
                                 width=6,
                                 minheight=2.5,
                                 vscale=1,
                                 imp_range1=int(np.min(imp)-0.001),
                                 imp_range2=int(np.max(imp)+0.001),
                                 color='#37354c',
                                 bgcolor=None,  # seaborn uses '#F1F8FE'
                                 xtick_precision=2,
                                 title=(str("All var imp_")+ str(ylabel_text)))
        if save_fig == 1:  
            #save figure for later - EPS
            fig_path=os.path.join(results_path,(str("ALL influencers var imp_")+ str(ylabel_text) + '.eps'))          
            plt.savefig(fig_path, format='eps', dpi=1000, transparent=True)
            
             #save figure for later -PDF
#            fig_path=os.path.join(results_path,(str("ALL influencers var imp_")+ str(ylabel_text) + '.pdf'))
#            plt.savefig(fig_path, format='pdf', dpi=1000)           
            
#            pickle.dump(fig, open(,'FigureObject.fig.pickle', 'wb') # This is for Python 3 - py2 may need `file` instead of `open`

        imp_pos = pd.DataFrame()   
        if 'R_A_N_D_O_M' in list(imp):
            stop_at_MSE = np.where(imp.index=='R_A_N_D_O_M')
        else:
            stop_at_MSE =  np.where(imp.values > 0)
            
        if np.array(stop_at_MSE[0]).size>0:
            imp_pos = imp.iloc[:stop_at_MSE[0][-1]]
            
        if not imp_pos.empty:
            models_and_methods.plot_importances2(imp_pos, yrot=0,
                     label_fontsize=10,
                     width=6,
                     minheight=2.5,
                     vscale=1,
                     imp_range1=int(np.min(imp)-0.001),
                     imp_range2=int(np.max(imp)+0.001),
                     color='#37354c',
                     bgcolor=None,  # seaborn uses '#F1F8FE'
                     xtick_precision=2,
                     title=(str("Positive influencers var imp_")+ str(ylabel_text)))

        if save_fig == 1:  
            #save figure for later -EPS
            fig_path=os.path.join(results_path,(str("Positive influencers var imp_")+ str(ylabel_text) + '.eps'))
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
            plt.savefig(fig_path, format='eps', dpi=1000, transparent=True)
            
            #save figure for later -PDF
            
            fig_path=os.path.join(results_path,(str("Positive influencers var imp_")+ str(ylabel_text) + '.pdf'))
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42
            plt.savefig(fig_path, format='pdf', dpi=1000, transparent=True)            
            #
#            pickle.dump(fig, open(,'FigureObject.fig.pickle', 'wb') # This is for Python 3 - py2 may need `file` instead of `open`
            
            
        return imp,imp_pos
    
    def create_var_imp_tables(rNames, unique_categories, uniqueness_specifier,varCount):
        ###################     OOB/MSE      ##############
        varImp_MSE = pd.DataFrame(index=rNames, columns = unique_categories)
        varImp_MSE.columns = unique_categories
        
        varImp_MSENUMS = pd.DataFrame(index=rNames, columns = unique_categories)
        varImp_MSENUMS.columns= unique_categories
        
        varImp_MSE_pos = pd.DataFrame(index=rNames, columns = unique_categories)
        varImp_MSE_pos.columns = unique_categories
        
            ###################     IncNodePurity      ##############
        varImp_NodePurity = pd.DataFrame(index=rNames, columns = unique_categories)
        varImp_NodePurity.columns = unique_categories
        
        varImp_NodePurityNUMS = pd.DataFrame(index=rNames, columns = unique_categories)
        varImp_NodePurityNUMS.columns= unique_categories
        
        varImp_NodePurity_pos = pd.DataFrame(index=rNames, columns = unique_categories)
        varImp_NodePurity_pos.columns = unique_categories
        
        #%% fill important rows
        category_Count = len(unique_categories)
        cohortCounts = pd.DataFrame(np.zeros(category_Count))
        
        for d in  range(0, category_Count):
            cohCount = int(np.sum(1*(uniqueness_specifier == unique_categories[d])))
        #    cohCount = int(np.sum(1*(X_train.install_age == unique_categories[d])))
            cohortCounts.iloc[d] = cohCount
        
            varImp_MSE[d][varCount+2] = cohCount
            varImp_MSENUMS[d][varCount+2] = cohCount
            varImp_MSE_pos[d][varCount+2] = cohCount
        
            varImp_NodePurity[d][varCount+2] = cohCount
            varImp_NodePurityNUMS[d][varCount+2] = cohCount
            varImp_NodePurity_pos[d][varCount+2] = cohCount

        return varImp_MSE, varImp_MSENUMS, varImp_MSE_pos, varImp_NodePurity, varImp_NodePurityNUMS,varImp_NodePurity_pos
    
    def extract_positive_features(varImp_MSE, varImp_MSENUMS,varImp_MSE_pos, varImp_NodePurity,varImp_NodePurityNUMS,varImp_NodePurity_pos, category_Count,varCount):
               
        for d in range(0, category_Count):
            if 'R_A_N_D_O_M' in list(varImp_MSE.iloc[:varCount+1,d]):
                stop_at_MSE = np.where((varImp_MSE.iloc[:varCount+1,d])=='R_A_N_D_O_M')
                stop_at2_MSE =  np.where(varImp_MSENUMS.iloc[:stop_at_MSE[0][0],d] > 0)
            else:
                stop_at2_MSE =  np.where(varImp_MSENUMS.iloc[:varCount+1,d] > 0)
        
            if np.array(stop_at2_MSE[0]).size>0:
                varImp_MSE_pos.iloc[stop_at2_MSE[0],d] = varImp_MSE.iloc[stop_at2_MSE[0],d]
                
            if 'R_A_N_D_O_M' in list(varImp_NodePurity.iloc[:varCount+1,d]):
                stop_at_NodePurity = np.where((varImp_NodePurity.iloc[:varCount+1,d])=='R_A_N_D_O_M')
                stop_at2_NodePurity =  np.where(varImp_NodePurityNUMS.iloc[:stop_at_NodePurity[0][0],d] > 0)
            else:
                stop_at2_NodePurity =  np.where(varImp_NodePurityNUMS.iloc[:varCount+1,d] > 0)        
            
            if np.array(stop_at2_NodePurity[0]).size > 0:
                varImp_NodePurity_pos.iloc[stop_at2_NodePurity[0],d] = varImp_NodePurity.iloc[stop_at2_NodePurity[0],d]
        
        return varImp_MSE_pos,varImp_NodePurity_pos   
    
    def build_flat_lstm_model(optimizer,n_neurons_0,input_shape1):    
        model = Sequential([
            LSTM(n_neurons_0, input_shape=input_shape1,kernel_initializer='lecun_uniform', return_sequences=True, name="myInput"),
            Dropout(rate=0.2),
            LSTM(n_neurons_0, return_sequences=True),
            LSTM(n_neurons_0, return_sequences=False),
            Dropout(rate=0.2),
            Dense(1, name="myOutput"),
            Activation("linear"),
            ])
        model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
#    model.summary()
        return model
    
    def build_flat_unstructured_lstm_model(optimizer,n_neurons_0, input_shape1):    
        model = Sequential([
            LSTM(n_neurons_0, input_shape =input_shape1,kernel_initializer='lecun_uniform', return_sequences=True, name="myInput"),
            Dropout(rate=0.2),
            LSTM(n_neurons_0, return_sequences=True),
            LSTM(n_neurons_0, return_sequences=False),
            Dropout(rate=0.2),
            Dense(1, name="myOutput"),
            Activation("linear"),
            ])
        model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
#    model.summary()
        return model

    def build_lstm_model(optimizer,n_neurons_1,n_neurons_2,n_neurons_3,n_neurons_4,input_shape1):    
        model = Sequential([
            LSTM(n_neurons_1, input_shape=input_shape1, activation='relu', kernel_initializer='lecun_uniform', return_sequences=True, name="myInput"),
            Dropout(rate=0.2),
            LSTM(n_neurons_2, return_sequences=True),
            LSTM(n_neurons_3, return_sequences=False),
            LSTM(n_neurons_4, return_sequences=False),
            Dropout(rate=0.2),
            Dense(1, name="myOutput"),
            Activation('linear'),
            ])
        model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
#    model.summary()
        return model

    def build_generic_nn_model(n_imp_vars,sigma,n_neurons_1,n_neurons_2,n_neurons_3,n_neurons_4):
    
        # Placeholder
        X = tf.placeholder(dtype=tf.float32, shape=[None, n_imp_vars], name="myInput")
        Y = tf.placeholder(dtype=tf.float32, shape=[None])
        
        #    # Initializers
        weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
        bias_initializer = tf.zeros_initializer()
        
        # Hidden weights
        W_hidden_1 = tf.Variable(weight_initializer([n_imp_vars, n_neurons_1]))
        bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
        W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
        bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
        W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
        bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
        W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
        bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
        
        # Output weights
        W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
        bias_out = tf.Variable(bias_initializer([1]))
        
        # Hidden layer
        hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
        hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
        hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
        
        # Output layer (transpose!)
        out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out), name="myOutput")
        
        # Cost function
        mse = tf.reduce_mean(tf.squared_difference(out, Y))
        # Optimizer
        opt = tf.train.AdamOptimizer().minimize(mse)
        
        return X, Y, out, mse, opt   

    def build_generic_unstructured_nn_model(n_imp_vars,sigma,n_neurons_1,n_neurons_2,n_neurons_3,n_neurons_4):
    
        # Placeholder
        X = tf.placeholder(dtype=tf.float32, shape=[None, None])
#        X = tf.placeholder(dtype=tf.float32, shape=[None, None], name="myInput")
        Y = tf.placeholder(dtype=tf.float32, shape=[None])
        
        #    # Initializers
        weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
        bias_initializer = tf.zeros_initializer()
        
        # Hidden weights
        W_hidden_1 = tf.Variable(weight_initializer( shape=[None, n_neurons_1]))
        bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
        W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
        bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
        W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
        bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
        W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
        bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
        
        # Output weights
        W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
        bias_out = tf.Variable(bias_initializer([1]))
        
        # Hidden layer
        hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
        hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
        hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
        
        # Output layer (transpose!)
        out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out), name="myOutput")
        
        # Cost function
        mse = tf.reduce_mean(tf.squared_difference(out, Y))
        # Optimizer
        opt = tf.train.AdamOptimizer().minimize(mse)
        
        return X, Y, out, mse, opt   
       
    def pred_by_point(model, data):
    		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def pred_multi_seq(model, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
#        data=np.array(data).flatten()
        prediction_seqs = []
        for i in range(int((data.shape[2])/prediction_len)):
            print(i)
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                print(j)

                if isinstance(curr_frame, (np.ndarray, np.generic)):
                    print(curr_frame.shape)
                    predicted.append(model.predict(curr_frame.reshape(1,1,-1))[0,0])
                    curr_frame = curr_frame[1:]
                    curr_frame = np.insert(curr_frame,np.array( [window_size-2]), predicted[-1], axis=1)
                else:
                    predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
                    curr_frame = curr_frame[1:]
                    curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
                prediction_seqs.append(predicted)
        return prediction_seqs

    def pred_full_seq(model, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted   
    
class PimpViz(models_and_methods):
    """
    For use with jupyter notebooks, plot_importances returns an instance
    of this class so we display SVG not PNG.
    """
    def __init__(self):
        tmp = tempfile.gettempdir()
        self.svgfilename = f"{tmp}/PimpViz_{getpid()}.svg"
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.savefig(self.svgfilename, bbox_inches='tight', pad_inches=0, transparent=True)

    def _repr_svg_(self):
        with open(self.svgfilename, "r", encoding='UTF-8') as f:
            svg = f.read()
        plt.close()
        return svg

    def save(self, filename):
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)

    def view(self):
        plt.show()

    def close(self):
        plt.close()
