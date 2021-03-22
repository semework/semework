#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 20:02:09 2021

@author: mulugetasemework
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, \
    BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesClassifier, \
    VotingRegressor, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))

from dash.dependencies import Input, Output
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import timedelta
from IPython.core.display import display, HTML
from matplotlib.pyplot import figure
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from numpy import newaxis
from os import getpid
from pandas.plotting import parallel_coordinates
from pathlib import Path
from plotly.graph_objs import *
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from random import randint, seed
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import binned_statistic
from scipy.stats.stats import pearsonr
from sklearn import datasets
from sklearn import model_selection, tree, tree as sklearn_tree
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge,Lasso, ElasticNet
from sklearn.metrics import accuracy_score,mean_squared_error, median_absolute_error
from sklearn.metrics import matthews_corrcoef, r2_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve as roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler,StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from textwrap import wrap
from time import sleep
import base64
import collections
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import datetime
import dateutil.parser
import functools
import io
import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import numpy as np
import operator
import os
import pandas
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline
import pydotplus
import random
import re
import seaborn as sns
import statsmodels.api as sm
import sys

#%%
pio.renderers.default = 'browser'
sns.set(style="ticks")
init_notebook_mode(connected=True)
#%%

app = dash.Dash(__name__)
server = app.server

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colors = {
    'background': '#57414C', 'fig_background':'black','fig_color':'black',
    'text': '#57414C','panelborder':'#031670','main_page_background_color':'#0b0342',
}

border_radius = 10,
border_radius_panels = 20,
main_page_style={'width': '97%', 'display': 'inline-block',  'padding': '0 20',
                  "margin-left": "1.5%","margin-right": "1.5%",'background-image': 'radial-gradient(#63324c, #2f709e)',
                  "border":"2px blue solid",'height': '20%','border-radius':border_radius,
                                    'box-shadow':' 1px 2px 18px  #69453'}
graph_panel_style_left={'display': 'inline-block',  'width': '45%', 'padding': '0 20',
                        'background-image': 'radial-gradient(ellipse at top, #42474a, transparent)','color':'black',
                  "margin-left": "2%","margin-right": "5.5%",
                  "border":colors['panelborder'],'height': '20%','border-radius':35,
                  'box-shadow':' 1px 2px 18px  #888888',
                                    'box-shadow':' 1px 2px 18px  #888888'}

graph_panel_style_right={'display': 'inline-block', 'width': '45%',
                         'background-image': 'radial-gradient(ellipse at top, #42474a, transparent)','color':'black',
                  "border":colors['panelborder'],'height': '20%','border-radius':35,
                  'box-shadow':' 1px 2px 18px  #888888'}
graph_panel_style={'display': 'inline-block', 'width': '90%', 'backgroundColor':colors['fig_background'],
                   "margin-left": "5%",'border-radius':10,
                  "border":"2px blue solid",'height': '7%', 'color':colors['fig_color'],
                                    'box-shadow':' 1px 2px 18px  #888888'}

table_style={'width': '90%', 'display': 'inline-block',  'padding': '0 20', 'backgroundColor':colors['panelborder'],
                  "margin-left": "5%","margin-right": "5%",'color':'black','height': '300px', 'overflowY': 'auto',
                  "border":"2px blue solid", 'border-radius':border_radius,
                                    'box-shadow':' 1px 2px 18px  #888888'}

table_style_narrow={'width': '30%', 'display': 'inline-block',  'padding': '0 20',
                    'backgroundColor':colors['main_page_background_color'],
                  "margin-left": "2%","margin-right": ".5%",'color':'black','height': '100px', 'overflowY': 'auto',
                  # "border":"2px blue solid",
                  'border-radius':25,'background-image': 'radial-gradient(ellipse at top, #42474a, transparent)',
                                    'box-shadow':' 1px 2px 18px  #888888'}

table_style_narrow_high={'width': '30%', 'display': 'inline-block',  'padding': '0 20', 'backgroundColor':colors['main_page_background_color'],
                  "margin-left": "2%","margin-right": ".5%",'color':'black','height': '190px', 'overflowY': 'auto',
                  'background-image': 'radial-gradient(ellipse at top, #42474a, transparent)',
                  'border-radius':25,'vertical-align': 'middle',
                                    'box-shadow':' 1px 2px 18px  #888888'}

text_div_narrow={'width': '30%', 'display': 'inline-block',  'padding': '0 0', 'backgroundColor':'white',
                  "margin-left": "2%","margin-right": ".5%",'color':'black','height': '25px', 'overflowY': 'auto',
}

title_text_style={'fontSize':22, 'textAlign':'center','border-radius':35,'color':'white', 'borderWidth': '1px',
                              # 'borderStyle': 'dashed','bordercolor':'white',
                              'width': '25%',"margin-left": "37.25%",
                                    'box-shadow':' 1px 2px 18px  #888888'}

title_text_style_wide={'fontSize':22, 'textAlign':'center','border-radius':35,'color':'white', 'borderWidth': '1px',
                              # 'borderStyle': 'dashed','bordercolor':'white',
                              'width': '50%',"margin-left": "25%",
                                    'box-shadow':' 1px 2px 18px  #888888'}

title_dials={'fontSize':13,   'color':'white','display': 'inline-block',
             'vertical-align': 'top', 'margin-left': '1vw',
                              'width': '30%', }

title_text_style_figure_panels={'fontSize':20, 'textAlign':'center','border-radius':35,'color':'white', 'borderWidth': '1px',
                              'borderStyle': 'dashed','bordercolor':'white',
                                    'box-shadow':' 1px 2px 18px  #888888'}

analysis_div={'display': 'inline-block',  'width': '25%', 'padding': '0 20', 'backgroundColor':colors['panelborder'],'color':'black',
                  "margin-left": "2%","margin-right": "2.5%",
                  "border":colors['panelborder'],'height': '10%','border-radius':45,
                  'box-shadow':' 1px 2px 18px  #888888',
                                    'box-shadow':' 1px 2px 18px  #888888'}

drop_downs_div={'display': 'inline-block', 'width': '25%', 'backgroundColor':colors['panelborder'],'color':'black',
                  "border":colors['panelborder'],'height': '10%','border-radius':45, "margin-left": "2%","margin-right": "2.5%",
                  'box-shadow':' 1px 2px 18px  #888888'}

file_saving_div={'display': 'inline-block', 'width': '25%', 'backgroundColor':colors['panelborder'],'color':'black',
                  "border":colors['panelborder'],'height': '10%','border-radius':45,
                  'box-shadow':' 1px 2px 18px  #888888'}

drop_down_style_main = {'color':'black',"margin-left": "6%",'border-radius':25,'padding': '-150px -150px',
                             'width': '90%','height': '22px','align':' center','borderColor': 'red',
                             'textAlign': 'center','font-size': '110%',}

drop_down_style_left = {'color':'black',"margin-left": "4%",'width': '65%','height': '30px','border-radius':25,
                        'align':' left','textAlign': 'center', 'borderColor': 'red',#'box-shadow':' 1px 2px 18px  #888888',
                             'font-size': '110%','padding': '-10px -10px'}

drop_down_style_right = {'color':'black',"margin-left": "30%",'width': '65%','height': '10px','border-radius':25,
                        'align':' right','textAlign': 'center', 'borderColor': 'red',
                             'font-size': '110%',"margin-top": "-5.5%",'padding': '-10px -10px'}

drop_titile = {'color':'white','textAlign':"left",'fontSize':18,
                                  'justify':"center", 'align':"center", "margin-left": "0%",
                                    'width':'30%','height': '10px',
                                'padding': '-50px -150px'}


#%%
# %% important parameters to play with
sampling_percentage = 1 # how much of the data we would like to use,
# in fractions, upto "1", meaning use all
predict_binary = 0 # change to 1 if the column to be predicted should be binarized
test_percentage = 0.3 # how much of the data to use for testing
corr_thresh = 0.95# drop one correlated column
p_val = 0.05
fully_recovered_prop_thresh = 0.6 # we somehow decided here 60 %is good outcome
sparsity_thresh = .5 # any column with more than this percentage of redundancy is discarded
# i.e. it has too many redundant values
NaN_thresh = 0.95 # any column with  this percentage of Nans is discarded
# intentionally kept high for this exercise since the few rows
# with data in sparse columns are still important
save_csv = 1
save_fig = 1
do_over_all_var_importance = 1
plot_importance_ranks = 1

#%%
app.layout = html.Div([

            html.Div([
                 html.H3("Your Data Dashboard"),
                 ],

                 className="banner",
                             style={
                'width': '100%',
                'height': '40px',
                'textAlign': 'center',
                'margin': '5px',
                'color':'white',
                'box-shadow':' 1px 2px 18px  #888888',
                  },
                  ),
#### ----------  upload button  --------------------
            html.Div([
                    dcc.Upload(
                         id='datatable-upload',
                         children=html.Div
                            ([
                              ('Drag and Drop or '),
                             html.A('Select Files')
                             ],style={'fontSize':22, 'textAlign':'center','Align':'center',# "margin-left": "30%",
                                     'border-radius':35,'color':'white',
                                     'bodercolor':'white','background-image': 'radial-gradient(ellipse at center, #4d2d2b, transparent)'}),

                         style={
                             'justify':'center',
                             "margin-left": "35%",
                             'width': '30%',
                             'textAlign':'center',
                              'height': '40px',
                              'align':' center',
                              'borderWidth': '1px',
                              'borderStyle': 'dashed',
                              'borderColor': 'white',
                              'borderRadius': '5px',
                             'textAlign': 'center',
                             # 'margin': '2px',
                             'font-size': '70.5%',
                               },
                         ),

html.Div([html.A('About this app', href='/Users/mulugetasemework/Dropbox/Phyton/TDI/DASH/Heroku/MData_wiki.html',
                 target='_blank')],
                         style={
                              'justify':'center',
                              "margin-left": "35%",
                              'width': '30%',
                              'textAlign':'center',
                              'height': '30px',
                              'align':' center',
                              'textAlign': 'center',
                              'color':'white',
                              'font-size': '150%',
                                },),

    html.Label(("A simple tool to analayze your data. Upload your CSV file above, results show up in the 18 figure plots below.\n"),
                   style={"margin-left": "5%",'textAlign':'center', 'fontSize':20, 'color':'lightyellow',
                            'width': '90%',
                             'textAlign':'center',
                              'height': '20px',}),
                    ]),
        html.Br(),

      html.Div([dash_table.DataTable(id='analysis_txt'),
                 html.Label(("Analysis setup"),
                                style={'fontSize':18, 'color':'black',
                                          'textAlign':'center',}),

                       ],style=text_div_narrow),


      html.Div([dash_table.DataTable(id='Data_txt'),
                 html.Label(("Data"),
                                style={'fontSize':18, 'color':'black',
                                          'textAlign':'center',}),

                       ],style=text_div_narrow),

      html.Div([dash_table.DataTable(id='file_txt'),
                 html.Label(("File saving"),
                                style={'fontSize':18, 'color':'black',
                                          'textAlign':'center',}),

                       ],style=text_div_narrow),

    html.Div([dash_table.DataTable(id='analysis'),

                dcc.RadioItems(id='data_selection_radio',
                    options=[
                        {'label': 'Categorical', 'value': 'cat'},
                        {'label': 'Numerical', 'value': 'Num'},
                        {'label': 'Both', 'value': 'Both'}
                    ],
                    value='Both',
                    labelStyle={'display': 'inline-block','color':'white','margin-right':'10px','margin-right':'10px'}
                        ),

    html.Label("Percentage of data to use",
                    style={'fontSize':16, 'textAlign':'center','color':'white', 'height': '20px',}),#,'box-shadow':' 1px 2px 18px  #888888'

                    dcc.Slider(id='dataSlider',
                        min=0,
                        max=100,
                        marks={i: ' {} %'.format(i) for i in list(range(0,101,25))},
                        value=75,
                    )
                  ],style=table_style_narrow),



        html.Div([dash_table.DataTable(id='dropdowns'),

        html.H6(" \n target", style=drop_titile),

        dcc.Dropdown(
            id = 'dropdown_target', placeholder='Select target/label',
            options=[], style=drop_down_style_main),
                        html.H6("x-axis", style=drop_titile),
                dcc.Dropdown(
                    id = 'crossfilter-xaxis-column', placeholder='Select X-Axis',
                    options=[], style=drop_down_style_main),
                html.H6("y-axis", style=drop_titile),
            # html.Br(),

                dcc.Dropdown(
                    id = 'crossfilter-yaxis-column', placeholder='Select Y-Axis',
                    options=[], style=drop_down_style_main),
                  ],style=table_style_narrow_high),

        html.Div([dash_table.DataTable(id='files'),
                dcc.Checklist(
                    options=[
                        {'label': 'png', 'value': 'png'},
                        {'label': 'pdf', 'value': 'pdf'},
                        {'label': 'fig', 'value': 'fig'},
                        {'label': 'csv', 'value': 'csv'},
                    ],
                    value=['png', 'csv'],
                labelStyle={'display': 'inline-block','color':'white','margin-right':'10px' }

                ) ,

        html.Button('Save now', id='save_button',style={'display': 'inline-block','color':'white','margin-left':'60%','width':'35%',
                  'border-radius':25,'text-align':'center','background':'grey',  'margin-right':'10px','margin-right':'10px'},),
                  ],style=table_style_narrow),
        html.Br(),
        html.Br(),
        html.Br(),

    html.Label("Scatter and line plots",
                   style=title_text_style),

        html.Div([

    dcc.RadioItems(
        id='crossfilter-xaxis-type',
        options=[ {'label': i, 'value': i} for i in ['Linear', 'Log'] ],
        value='Linear',style={'color':'white'},
        labelStyle={'display': 'inline-block'}
                  ),

            dcc.Graph(
                id='crossfilter-indicator-scatter',style=graph_panel_style,
            )

        ], style=graph_panel_style_left),

        html.Div([
        dcc.RadioItems(
            id='crossfilter-yaxis-type',style={'color':'white'},
            options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
            value='Linear',
            labelStyle={'display': 'inline-block'}
                    ),
                dcc.Graph(id='x-time-series',style={'display': 'inline-block','margin-left':'10%', 'width': '80%','height':'225px'}),
                dcc.Graph(id='y-time-series',style={'display': 'inline-block', 'margin-left':'10%','width': '80%','height':'225px'}),
        ], style=graph_panel_style_right),
        html.Br(),
        html.Br(),

    ######### -------       corr      RWW 2    --------
    ######### -------       corr      RWW 2    --------
    ######### -------       corr      RWW 2    --------

    html.Label("Correlation plots",
                   style=title_text_style),

        html.Div([
            dcc.Graph(
                id='correlationPlot',style=graph_panel_style,
            ),
        ], style=graph_panel_style_left),

        html.Div([
            dcc.Graph(id='corr_network',style=graph_panel_style),
        ], style=graph_panel_style_right),
        html.Br(),
        html.Br(),

    ######### -------       Pairplots      RWW 3    --------
    ######### -------       Pairplots      RWW 3    --------
    ######### -------       Pairplots      RWW 3    --------

    html.Label("Pairplots and distributions",
                   style=title_text_style),

        html.Div([
            dcc.Graph(
                id='pairplots',style=graph_panel_style,
                # hoverData={'points': [{'customdata': 'Japan'}]}
            )
        ], style=graph_panel_style_left),

        html.Div([
            dcc.Graph(id='distributions',style=graph_panel_style),
        ], style=graph_panel_style_right),
        html.Br(),
        html.Br(),

    ######### -------       Variable importance      RWW 4    --------
    ######### -------                                RWW 4   --------
    ######### -------                                RWW 4    --------

    html.Label("Variable importance",
                   style=title_text_style),

        html.Div([
            dcc.Graph(
                id='variable_importance_bar',style=graph_panel_style,
            )
        ], style=graph_panel_style_left),

        html.Div([
            dcc.Graph(id='variable_importance_wordcloud',style=graph_panel_style),
        ], style=graph_panel_style_right),

        html.Br(),
        html.Br(),
    ######### -------       PCA      RWW 5    --------
    ######### -------                RWW 5    --------
    ######### -------                 RWW 5    --------

    html.Label("PCA",
                   style=title_text_style),

        html.Div([
            dcc.Graph(
                id='PCA_plot',style=graph_panel_style,
                # hoverData={'points': [{'customdata': 'Japan'}]}
            )
        ], style=graph_panel_style_left),

        html.Div([
            # dcc.Graph(id='x-time-series'),
            dcc.Graph(id='PCA_network',style=graph_panel_style),
        ], style=graph_panel_style_right),
        html.Br(),

        html.Br(),
    ######### -------      RF n dendrogram      RWW 6    --------
    ######### -------                           RWW 6    --------
    ######### -------                           RWW 6    --------

    html.Label("Trees and dendrograms",
                   style=title_text_style),

        html.Div([
            dcc.Graph(
                id='RF_plot',style=graph_panel_style,
                # hoverData={'points': [{'customdata': 'Japan'}]}
            )
        ], style=graph_panel_style_left),

        html.Div([
            # dcc.Graph(id='x-time-series'),
            dcc.Graph(id='dendrogram',style=graph_panel_style),
        ], style=graph_panel_style_right),

        html.Br(),
        html.Br(),

    ######### -------      Network      RWW 7    --------
    ######### -------                   RWW 7    --------
    ######### -------                   RWW 7    --------

    html.Label(" Network plots",
                   style=title_text_style),

        html.Div([
            dcc.Graph(
                id='network',style=graph_panel_style,
                # hoverData={'points': [{'customdata': 'Japan'}]}
            )
        ], style=graph_panel_style_left),

        html.Div([
            # dcc.Graph(id='x-time-series'),
            dcc.Graph(id='network_interactive',style=graph_panel_style),
        ], style=graph_panel_style_right),


        html.Br(),
        html.Br(),



    ######### -------      performance whisker and ROC     RWW 8    --------
    ######### -------                                      RWW 8    --------
    ######### -------                                      RWW 8    --------

    html.Label('Model performances',
                   style=title_text_style),

        html.Div([
            dcc.Graph(
                id='performance_whisker',style=graph_panel_style,
            )
        ], style=graph_panel_style_left),

        html.Div([
            dcc.Graph(id='roc',style=graph_panel_style),
        ], style=graph_panel_style_right),

        html.Br(),
        html.Br(),


    ######### -------       performance and predictions      RWW 9    --------
    ######### -------                                        RWW 9    --------
    ######### -------                                        RWW 9    --------

    html.Label("Model performance (precision, recall) and predictions by best model",
                   style=title_text_style_wide),

        html.Div([
            dcc.Graph(
                id='precision_recall',style=graph_panel_style,
            )
        ], style=graph_panel_style_left),

        html.Div([
            dcc.Graph(id='predictions',style=graph_panel_style),
        ], style=graph_panel_style_right),

        html.Br(),
        html.Br(),

    html.Label("Your data",
                   style=title_text_style),

    html.Div([dash_table.DataTable(id='datatable-upload-container'),],style=table_style),


    ], style=main_page_style)

#%%
# ((((((($($%($$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$)))))))))
# ((((((($($%($$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$)))))))))

#%%
def RF_predict(df1, target):
    rf = RandomForestRegressor(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               random_state = 42)


    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
    # categorical_features = df1.select_dtypes(include=['object', 'bool' ]).columns
    str_features = df1.select_dtypes(include=[ 'object','string']).columns

    for i in str_features:
        df1[i] = df1[i].astype('category').cat.codes
        dfv = pd.get_dummies(df1, columns=[i], prefix=str(i)+'_oh')

    colNames = dfv.columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude=object)),
        ('cat', categorical_transformer, selector(dtype_include=object))
    ])
    y = dfv[[target]].reset_index()
    Xv =  dfv.drop(target, axis=1).reset_index()

    colNames =  Xv.columns
    X = preprocessor.fit_transform(Xv)
    X_train, X_test, y_train, y_test  =  \
            train_test_split(X, y,  test_size = 0.2, random_state=12345)

    rf_fitted = rf.fit(X_train, y_train)
    X_train = pd.DataFrame(X_train)
    X_train.columns = colNames
    return rf_fitted, X_train.columns

#%%
def clean_up_data(All_data, sparsity_thresh, NaN_thresh, corr_thresh):
    indexer_errors = ['level_0','index']
    for i in indexer_errors:
        if i in All_data.columns:
            All_data = All_data.drop(i, axis=1)

    All_data = All_data.loc[:,~All_data.columns.duplicated()]
    All_data = All_data.dropna(axis=1, how='all')
    All_data = All_data.dropna(axis=0, how='all')
    # All_data = All_data.loc[:, (All_data != All_data.iloc[0]).any()]
    All_data = All_data.reset_index(drop=True)


    # first  clean up data with sparsity analysis
    # remove columns which are very sparsly populated as they might cause false results
    # such as becoming very important in predictions despite having few real data points
    # column 1 for this data is ID, so it can be repeated
    sparse_cols = [((len(All_data.iloc[:,i].unique())/len(All_data))*100)  <  (int((1-sparsity_thresh)*len(All_data)))
                   for  i  in range(0, All_data.shape[1] )]

    #remove sparse columns (i.e. with too many redundant values)
    All_data = All_data.iloc[:, sparse_cols]

    #remove too-many NaN columns
    non_NaN_cols = [All_data.iloc[:,i].isna().sum() < int(NaN_thresh*len(All_data)) for i in range(All_data.shape[1])]
    All_data = All_data.iloc[:, non_NaN_cols]

    # drop the pesky "Unnamed: 0" column, if exists
    # this happens sometimes depending on which packes are used or the quality of the .CSV file

    unNamedCols =   All_data.filter(regex='Unnamed').columns

    if not unNamedCols.empty:
        for i in unNamedCols:
            if i in All_data.columns:
                All_data = All_data.drop(i, axis=1)
        # All_data4 = All_data.drop(unNamedCols), axis=1, inplace=True)

    # drop highly correlated columns
    cols = All_data.select_dtypes([np.number]).columns
        # to dot that, first Create correlation matrix
    corr_matrix = All_data.reindex(columns=cols).corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column  for column in upper.columns if any(upper[column] > corr_thresh)]
    # Drop Marked Features
    All_data.drop(All_data[to_drop], axis=1)

    return All_data
#%%

###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                     corr
def do_corr(df, imp_col=[]):
    # corr matrix
    if len(imp_col)> 0:
        imp_col = imp_col
    else:
        imp_col = df.columns[-1]

    corr = df.convert_dtypes().corr(method ='pearson')
    corr = corr[corr.columns[::-1]]
    corr = corr.sort_values(by=imp_col, ascending=[False ])
    colls = corr.index

    return corr, colls

def plot_corrs(corr, imp_col=[]):
    if len(imp_col)> 0:
        imp_col = imp_col
    else:
        imp_col = corr.columns[-1]
    # lower triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.close("all")
    plt.figure(figsize=(12,10), dpi= 80)
    # colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title("Correlogram (correlation statistics)"
               "\n"
              '(coefficients sorted by ' + str(imp_col) + ' (left column)). \n'
               'Order in left vertical axis shows correlation with ' + str(imp_col) + ' \n'
               '(i.e. Next to ' + str(imp_col) + ' themselves, top parameters'
                'are highly correlated with ' + str(imp_col) + ')',
              fontsize=14)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.margins(0.01,0.01)
    plt.show()

#%%
def scale_data(X_train,X_test,y_train,y_test):


#        scaler_mat = Normalizer()
#        #scaler for test/train array, model will be returned for inverse transform
#        scaler_y = Normalizer()
#

#        scaler_mat = RobustScaler()
#        #scaler for test/train array, model will be returned for inverse transform
#        scaler_y = RobustScaler()
#
    scaler_mat = StandardScaler()
    scaler_y = StandardScaler()
#        #scaler for matrices
#        scaler_mat = MinMaxScaler()
#        #scaler for test/train array, model will be returned for inverse transform
#        scaler_y = MinMaxScaler()
#
    data_train = pd.DataFrame(X_train.reset_index(drop=True))
    data_test = pd.DataFrame(X_test.reset_index(drop=True))
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

#%%
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

    imp,rf = permutation_importances_raw(rf, X_train, y_train, metric, n_samples)
    I = pd.DataFrame(data={'Feature':X_train.columns, 'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I,rf

def permutation_importances_raw(rf, X_train, y_train, metric, n_samples=-1):
    """
    Return array of importances from pre-fit rf; metric is function
    that measures accuracy or R^2 or similar. This function
    works for regressors and classifiers.
    """

    X_sample, y_sample = sample(X_train, y_train, n_samples)

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
    return np.array(imp),rf

#%%
def over_all_var_importance(predict_binary,ylabel_text,save_fig,results_path,X_train,y_train,plot_importance_ranks):

    RANDOM_STATE = 999

    if predict_binary == 1:
        rf = RandomForestClassifier(n_estimators=100,  warm_start=True,
     # better generality with 5
     min_samples_leaf=5, n_jobs=-1, oob_score=True, max_features="sqrt",
                           random_state=RANDOM_STATE)
    else:
        rf = RandomForestRegressor(n_estimators=100,random_state=RANDOM_STATE, n_jobs=-1, oob_score=True)

    o_X = X_train.copy().astype(np.int64)
    o_y = y_train.copy().astype(np.int64)

    if predict_binary == 0:
        o_X = X_train.copy().astype(np.float)
        o_y = y_train.copy().astype(np.float)

    rf.fit(o_X,o_y)

    if isinstance(rf, RandomForestClassifier):
       imp,rf =  permutation_importances2(rf, pd.DataFrame(o_X), pd.DataFrame(o_y),
                                  oob_classifier_accuracy)
    elif isinstance(rf, RandomForestRegressor):
       imp,rf =  permutation_importances2(rf,pd.DataFrame(o_X), pd.DataFrame(o_y),
                                   oob_regression_r2_score)


    imp_pos = pd.DataFrame()
    if 'R_A_N_D_O_M' in list(imp.index):
        stop_at_MSE = min(np.where(imp.values > 0)[0][-1], np.where(imp.index=='R_A_N_D_O_M')[0][-1])
    else:
        stop_at_MSE =  np.where(imp.values > 0)[0][-1]

    if np.array(stop_at_MSE).size>0:
        imp_pos = imp.iloc[:stop_at_MSE+1]


    print('rf is: ', rf)

    return imp, imp_pos, rf
#%% var imp

def var_imp(df1, target):

    X = df1[[df1.columns != target]]

    y = df1[[target]]


    X_train, X_test, y_train, y_test  =  \
            train_test_split(X, y,  test_size = 0.4, random_state=12345)
    colNames =  X_train.columns
    X_train, X_test, y_train, y_test, scaler_y =  scale_data(X_train, X_test, y_train, y_test)

    y_test = pd.DataFrame(y_test.astype(np.int64))
    y_train = pd.DataFrame(y_train.astype(np.int64))

    # put column names back
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    X_train.columns = colNames

    X_test.columns = colNames

    X_train = X_train.assign(R_A_N_D_O_M=np.random.random(size=len(y_train)))
    X_test= X_test.assign(R_A_N_D_O_M=np.random.random(size=len(y_test)))

    imp, imp_pos, rf =  over_all_var_importance(predict_binary,X_train,y_train,plot_importance_ranks)
    return imp_pos


#%%   pair and freq
def parse_contents(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')) )
            elif 'xlsx' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded)  )

        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        unNamedCols =   df.filter(regex='Unnamed').columns

        if not unNamedCols.empty:
            for i in unNamedCols:
                if i in df.columns:
                    df  = df.drop(i, axis=1)

        df = df.convert_dtypes()

        return df
    else:
        return [{}]

@app.callback(Output('datatable-upload-container', 'data'),
              Output('datatable-upload-container', 'columns'),
              Input('datatable-upload', 'contents'),
              State('datatable-upload', 'filename'))
def update_output(contents, filename):
    if contents is None:
        return [{}], []
    df_uploaded = parse_contents(contents, filename)
    # df_uploaded = clean_up_data(df_uploaded , sparsity_thresh, NaN_thresh, corr_thresh)

    # process_pipeline(df)
    return df_uploaded.to_dict('records'), [{"name": i, "id": i} for i in df_uploaded.columns]

# update y-dropdown
@app.callback(Output('crossfilter-yaxis-column', 'options'),
              [Input('datatable-upload', 'contents'),
                Input('datatable-upload', 'filename'),
                Input("data_selection_radio", "value"),
                    ])

def update_y_dropdown(contents, filename, data_selection ):
    if contents is not None:
        df1 = parse_contents(contents, filename)

        if data_selection == 'None':
            df1 = df1
        elif data_selection == 'Categorical':
            df1 = df1.select_dtypes(['category'])
        elif data_selection == 'Numerical':
            df1 = df1.select_dtypes(include=np.number)
        else:
            df1 = df1
        df1 =  clean_up_data(df1, sparsity_thresh, NaN_thresh, corr_thresh)

        columns = df1.columns.values.tolist()
        if df1 is not None:
            return [ {'label': x, 'value': x} for x in columns ]
        else:
            return []
    else:
        return []

# update x-dropdown
@app.callback(Output('dropdown_target', 'options'),
              [Input('datatable-upload', 'contents'),
               Input('datatable-upload', 'filename'),
                   Input("data_selection_radio", "value"),])
def update_target_dropdown(contents, filename,data_selection ):
    if contents is not None:
        df1 = parse_contents(contents, filename)

        if data_selection == 'None':
            df1 = df1
        elif data_selection == 'Categorical':
            df1 = df1.select_dtypes(['category'])
        elif data_selection == 'Numerical':
            df1 = df1.select_dtypes(include=np.number)
        else:
            df1 = df1

        columns = df1.columns.tolist()
        if df1 is not None:
            return [ {'label': x, 'value': x} for x in columns ]
        else:
            return []
    else:
        return []


# update x-dropdown
@app.callback(Output('crossfilter-xaxis-column', 'options'),
              [Input('datatable-upload', 'contents'),
               Input('datatable-upload', 'filename'),
                   Input("data_selection_radio", "value"),])


def update_x_dropdown(contents, filename,  data_selection ):

    if contents is not None:

        df1 = parse_contents(contents, filename)

        if data_selection == 'None':
            df1 = df1
        elif data_selection == 'Categorical':
            df1 = df1.select_dtypes(['category'])
        elif data_selection == 'Numerical':
            df1 = df1.select_dtypes(include=np.number)
        else:
            df1 = df1


        columns = df1.columns.tolist()

        if df1 is not None:
            return [ {'label': x, 'value': x} for x in columns ]
        else:
            return []
    else:
        return []


def data_processing(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)
    #df1 = df1.convert_dtypes()
    df1 = df1.sample(frac=datasize/100)


    if data_selection == 'None':
        df1 = df1
    elif data_selection == 'Categorical':
        df1 = df1.select_dtypes(['category'])
    elif data_selection == 'Numerical':
        df1 = df1.select_dtypes(include=np.number)
    else:
        df1 = df1
    df1 =  clean_up_data(df1, sparsity_thresh, NaN_thresh, corr_thresh)

    return df1

#################################################################################
#                crossfilter-indicator-scatter
#################################################################################
@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('datatable-upload-container', 'data'),
    Input('dropdown_target', 'value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input("data_selection_radio", "value"),
    Input("dataSlider", "value"),
    prevent_initial_call=True
    )
def update_scat(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)

    df1 = data_processing(df1, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize)

    try:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df1[xaxis_column_name], y=df1[target],
            name='sin',
            mode='markers',
            marker_color='rgba(152, 0,100, .8)'
        ))

        # Set options common to all traces with fig.update_traces
        fig.update_traces(mode='markers', marker_line_width=3, marker_size=13)
        fig.update_layout(title='Scatter',
                          yaxis_zeroline=False, xaxis_zeroline=False)

        fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

        fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

        fig.update_layout(margin={'l': 40, 'b': 40, 't': 20, 'r': 10}, hovermode='closest')


    except:
        fig = {
            'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
    return fig


#################################################################################



def create_time_series(dff, axis_type, ydata, title, target):

    fig = px.scatter(dff, x=ydata, y=target)
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    Input('datatable-upload-container', 'data'),
    Input('dropdown_target', 'value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input("data_selection_radio", "value"),
    Input("dataSlider", "value"),
    prevent_initial_call=True
    )

def update_y_timeseries(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)


    df1 = data_processing(df1, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize)

    if (df1.empty or len(df1.columns) < 1):
        df1 = {
            'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
        # xaxis_type, xaxis_column_name, yaxis_column_name, title = 'Linear','Linear','',''
    title = '<b>{}</b><br>{}'.format(target, yaxis_column_name)
    return create_time_series(df1, yaxis_type, yaxis_column_name, title, target)

@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    Input('datatable-upload-container', 'data'),
    Input('dropdown_target', 'value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input("data_selection_radio", "value"),
    Input("dataSlider", "value"),
    prevent_initial_call=True
    )
def update_x_timeseries(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)
    df1 = data_processing(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize)

    if (df1.empty or len(df1.columns) < 1):
        df1 = {'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
        # xaxis_type, xaxis_column_name, yaxis_column_name, title = 'Linear','Linear','',''

    title = '<b>{}</b><br>{}'.format(target, xaxis_column_name)
    return create_time_series(df1, xaxis_type, xaxis_column_name, title, target)



#################################################################################
#                correlationPlot
#################################################################################

@app.callback(
    Output('correlationPlot', 'figure'),
    Input('datatable-upload-container', 'data'),
    Input('dropdown_target', 'value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input("data_selection_radio", "value"),
    Input("dataSlider", "value"),
    prevent_initial_call=True
    )
def update_corr_heatmap(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)

    try:

        # corr =  df1.corr(method='pearson')
        corr, colls =  do_corr(df1, imp_col=target)
        # corr =  df1.corr(method='pearson')
        fig = px.imshow(corr)

    except:
        try:
            corr, colls =  do_corr(df1, imp_col=target)
            fig = px.imshow(corr)

        except:
            fig = {
                'data': [{
                    'x': [],
                    'y': [],
                    'type': 'scatter'
                }]
            }
    return fig


@app.callback(
    Output('distributions', 'figure'),
    Input('datatable-upload-container', 'data'),
    Input('dropdown_target', 'value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input("data_selection_radio", "value"),
    Input("dataSlider", "value"),
    prevent_initial_call=True
    )
def update_correlation_d_plot(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)

    try:

        fig = px.box(df1)
        return fig
    except:
        try:
            fig = px.histogram(df1, x=yaxis_column_name,
                                   histnorm='percent',
                                   nbins = 10)
        except:
            fig = {
                'data': [{
                    'x': [],
                    'y': [],
                    'type': 'scatter'
                }]
            }
    return fig

@app.callback(
    dash.dependencies.Output('pairplots', 'figure'),
    Input('datatable-upload-container', 'data'),
    Input('dropdown_target', 'value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input("data_selection_radio", "value"),
    Input("dataSlider", "value"),
    prevent_initial_call=True
    )

def update_pairplot_grpah1(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)

    if (df1.empty or len(df1.columns) < 1):
        return {
                'data': [{
                    'x': [],
                    'y': [],
                    'type': 'bar'
                }]
            }

    try:
        fig = ff.create_scatterplotmatrix(df1)
        return fig
    except:
        try:
            fig = ff.create_scatterplotmatrix(df1)
        except:
            fig = {
                'data': [{
                    'x': [],
                    'y': [],
                    'type': 'scatter'
                }]
            }
    return fig



# Create graph component and populate with scatter plot
@app.callback(
    Output('performance_whisker', 'figure'),
    Input('datatable-upload-container', 'data'),
    Input('dropdown_target', 'value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input("data_selection_radio", "value"),
    Input("dataSlider", "value"),
    prevent_initial_call=True
    )

# def update_grpah(selected_counties, selected_state):
def update_pairplot_whisker1(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)
    if (df1.empty or len(df1.columns) < 1):
        return {
                'data': [{
                    'x': [],
                    'y': [],
                    'type': 'bar'
                }]
            }

    try:
        fig = px.box(df1, y=target)
    except:
        fig = {
            'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
    return fig



#################################################################################
#               variable_importance_bar
#################################################################################
@app.callback(Output('variable_importance_bar', 'figure'),
    Input('datatable-upload-container', 'data'),
    Input('dropdown_target', 'value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input("data_selection_radio", "value"),
    Input("dataSlider", "value"),
    prevent_initial_call=True
    )

def display_bargraph1(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)

    if (df1.empty or len(df1.columns) < 1):
        return {
                'data': [{
                    'x': [],
                    'y': [],
                    'type': 'bar'
                }]
            }

    try:

        df1 = data_processing(df1, target, hoverData, xaxis_column_name, yaxis_column_name,
                    xaxis_type, yaxis_type, data_selection, datasize)

        rf, colN = RF_predict(df1, target)
        x = rf.feature_importances_
        # y = colN
        fi = pd.DataFrame({'feature': colN,
                           'importance': x}).\
                            sort_values('importance', ascending = False)

        fig = go.Figure(go.Bar(
                    x=fi.importance,
                    y=fi.feature,
                    orientation='h'))
    except:

        fig = {
            'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
    return fig


@app.callback(Output('variable_importance_wordcloud', 'figure'),
    Input('datatable-upload-container', 'data'),
    Input('dropdown_target', 'value'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-yaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'),
    Input('crossfilter-yaxis-type', 'value'),
    Input("data_selection_radio", "value"),
    Input("dataSlider", "value"),
    prevent_initial_call=True
    )

def variable_importance_wordcloud_graph(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)
    if (df1.empty or len(df1.columns) < 1):
        return {
            'data': [{
                'x': [],
                'y': [],
                'type': 'pie'
            }]
        }
    else:
        thisFig ={
        'data': [{
            'x': df1[df1.columns[0]] ,
            'y': df1[df1.columns[0]],
            'type': 'pie','title':'Variable importance',
        }]
    }

    return thisFig

if __name__ == '__main__':
    app.run_server(debug=True)