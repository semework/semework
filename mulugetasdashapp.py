from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, \
    BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesClassifier, \
    VotingRegressor, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import (accuracy_score,mean_squared_error, median_absolute_error,
                             matthews_corrcoef, r2_score, confusion_matrix, roc_auc_score,
                             classification_report, r2_score, roc_curve as roc_curve)

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import timedelta
from IPython.core.display import display, HTML
from IPython.display import Image
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
from plotly.offline import plot
from random import randint, seed
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import binned_statistic
from scipy.stats.stats import pearsonr
from sklearn import datasets, model_selection, tree
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge,Lasso, ElasticNet
from sklearn.model_selection import (cross_val_score, KFold, train_test_split, GridSearchCV)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from wordcloud import WordCloud
import base64
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import io
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline
import random
import seaborn as sns
from sklearn import tree


def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples,
                                              n_samples_bootstrap)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices

#%%
pio.renderers.default = 'browser'
sns.set(style="ticks")
init_notebook_mode(connected=True)
#%%

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title='Mcapstone'
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

#%%  vars for popuiating fields before loading data


(target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize)=[],[],[],[],'linear','linear','Both', 75

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

#%%
#### ----------  app  --------------------------------------------------------
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
#### ----------  upload button  ----------------------------------------------
            html.Div([


                    dcc.Upload(
                         id='datatable-upload',
                         children=html.Div
                            ([
                              ('Drag and Drop or '),
                             html.A('Select Files')
                             ],style={'fontSize':22, 'textAlign':'center','Align':'center',# "margin-left": "30%",
                                     'border-radius':35,'color':'white',
                                     'bodercolor':'white','background-image':
                                         'radial-gradient(ellipse at center, #4d2d2b, transparent)'}),

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


#### ----------  about this app ----------------------------------------------


html.Div([html.A('About this app', href='https://github.com/semework/semework',
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
#### ----------  description -------------------------------------------------
     html.Label(("A simple tool to analayze your data. Upload your CSV file above, results show up in the 18 figure plots below.\n"),
                   style={"margin-left": "5%",'textAlign':'center',
                          'fontSize':20, 'color':'lightyellow',
                            'width': '90%',
                             'textAlign':'center',
                              'height': '20px',}),
                    ]),
      html.Br(),

#### ----------  analysis setup -----------------------------------------------

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

#### ----------  dropdowns     -----------------------------------------------

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

            dcc.Dropdown(
                id = 'crossfilter-yaxis-column', placeholder='Select Y-Axis',
                options=[], style=drop_down_style_main),
              ],style=table_style_narrow_high),

#### ----------  file saving   -----------------------------------------------

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

        ######### -------       scatter      RWW 1    --------

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

    html.Label("Variable importance",
                   style=title_text_style),

        html.Div([

        html.Div([html.A('what is variable importance?', href='https://github.com/semework/semework/blob/main/RF%20&%20variable%20importance.md',
                 target='_blank')],
                         style={
                              'justify':'center',
                              "margin-left": "35%",
                              'width': '70%',
                              'textAlign':'center',
                              'height': '30px',
                              'align':' center',
                              'textAlign': 'center',
                              'color':'white',
                              'font-size': '100%',
                                },),


            dcc.Graph(
                id='variable_importance_bar',style=graph_panel_style,
            ),

        ], style=graph_panel_style_left),

        html.Div([
            dcc.Graph(id='variable_importance_wordcloud',style=graph_panel_style),
        ], style=graph_panel_style_right),

        html.Br(),
        html.Br(),


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

    ######### -------       PCA      RWW 5    --------


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


    html.Label("Trees and dendrograms",
                   style=title_text_style),

        html.Div([
            dcc.Graph(
                id='RF_plot',style=graph_panel_style,
            )
        ], style=graph_panel_style_left),

        html.Div([
            # dcc.Graph(id='x-time-series'),
            dcc.Graph(id='dendrogram',style=graph_panel_style),
        ], style=graph_panel_style_right),

        html.Br(),
        html.Br(),

    ######### -------      Network      RWW 7    --------


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
def RF_predict(X,  y):

    rf = RandomForestRegressor(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               random_state = 42)


    X_train, X_test, y_train, y_test1  =  \
            train_test_split(X, y, test_size = 0.2, random_state=42)

    rf_fitted_all = rf.fit(X_train, y_train)

    X_train = pd.DataFrame(X_train)


    y_test = np.array(y_test1).reshape(-1,1)

    x = rf_fitted_all.feature_importances_

    fi = pd.DataFrame({'feature': X_train.columns,
                       'importance': x}).\
                        sort_values('importance', ascending = True)

    y_pred_all = rf_fitted_all.predict(X_test)

    rf_fitted_imp = rf.fit(np.array(X_train[fi.feature[0]]).reshape(-1, 1), y_train)

    y_pred_best = rf_fitted_imp.predict(y_test)


    perf_all = return_accuracies(X_train, X_test, y_train, y_test, y_pred_all)
    Xtr ,xts = np.array(X_train[fi.feature[0]]).reshape(-1, 1), np.array(X_test[fi.feature[0]]).reshape(-1, 1)

    perf_best = return_accuracies(Xtr ,xts, y_train, y_test, y_pred_best)

    perf = pd.concat([perf_all, perf_best],axis=1)

    perf.columns = ['All features','Most important predector']
    perf.index = 'r2_Score','adj_r2_score','MSE', 'MAE'

    return  rf_fitted_all, y_pred_all, y_pred_best, y_test1, fi, fi.feature[0], perf

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

    scaler_mat = StandardScaler()
    scaler_y = StandardScaler()

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

def r2_scores(y_true, y_pred):
    return r2_score(y_true, y_pred)

def adj_r2_score(r2, n, k):   # k stands for the number of predictors (regressors), n is sample size (row count)

    ''' example models_and_methods.adj_r2_score(r2_test, X_test1.shape[0], X_test1.shape[1])) '''

    if (n-k-1) != 0:
        r2sc = 1-((1-r2)*((n-1)/(n-k-1)))
    else:
        r2sc = 0
    return r2sc

def mse1(y_true, y_pred):
    return mean_squared_error(y_true.ravel(), np.array(y_pred).ravel())

def mae1(y_true, y_pred):
    return median_absolute_error(y_true.ravel(), np.array(y_pred).ravel())


#%%
def return_accuracies(X_train, X_test, y_train, y_test, y_pred):
            ##%% 1 r2score
    R2Score =   r2_scores(y_test, y_pred)

    #%% 2
    adjr2 =   adj_r2_score(R2Score, X_test.shape[0], X_test.shape[1])

    #%% 3
    MSE =  mse1(y_test, y_pred)

    #%% 4  median absolute error

    MAE =  mae1(y_test, y_pred)


    return pd.DataFrame([R2Score, adjr2, MSE, MAE ])
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


    unNamedCols =   All_data.filter(regex='Unnamed').columns

    if not unNamedCols.empty:
        for i in unNamedCols:
            if i in All_data.columns:
                All_data = All_data.drop(i, axis=1)

    # drop highly correlated columns
    cols = All_data.select_dtypes([np.number]).columns
        # to dot that, first Create correlation matrix
    corr_matrix = All_data.reindex(columns=cols).corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    to_drop = [column  for column in upper.columns if any(upper[column] > corr_thresh)]
    # Drop Marked Features
    All_data.drop(All_data[to_drop], axis=1)

    return All_data

def data_processing(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):

    df1 = pd.DataFrame(rows)
    df1 = df1.convert_dtypes()
    df1 = df1.sample(frac=datasize/100)

    numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
    str_features = df1.select_dtypes(include=[ 'object','string']).columns


    if data_selection == 'None':
        df1 = df1
    elif data_selection == 'Categorical' and len(str_features) > 0:
        df1 = df1[[str_features]]
    elif data_selection == 'Numerical'  and len(numeric_features) > 0:
        df1 = df1[[numeric_features]]
    else:
        df1 = df1

    df1 =  clean_up_data(df1, sparsity_thresh, NaN_thresh, corr_thresh)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor1 = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude=object)),
        ('cat', categorical_transformer, selector(dtype_include=object))
    ])
    numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns

    str_features = df1.select_dtypes(include=[ 'object','string']).columns

    cols = df1.columns

    if len(str_features) > 0:
        for i in str_features:
            df1[i] = df1[i].astype('category').cat.codes

    dfv = df1.copy()


    df1 = pd.DataFrame(preprocessor1.fit_transform(dfv))

    return df1, str_features, numeric_features, cols

def data_transformation(df1, target):

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor1 = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude=object)),
        ('cat', categorical_transformer, selector(dtype_include=object))
    ])

    str_features = df1.select_dtypes(include=[ 'object','string']).columns

    if len(str_features) > 0:
        for i in str_features:
            df1[i] = df1[i].astype('category').cat.codes
            dfv = pd.get_dummies(df1, columns=[i], prefix=str(i)+'_oh')
    else:
        dfv = df1.copy()

    y = pd.DataFrame(dfv[[target]].values).reset_index()[0]

    Xv =  dfv.drop(target, axis=1).reset_index()

    colNames = Xv.columns

    colNames = colNames[colNames != 'index']

    Xv = Xv[colNames]

    rows = pd.DataFrame(preprocessor1.fit_transform(Xv))

    rows.columns = colNames

    return rows, y, colNames

#%%   pair and freq
def parse_contents(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')) )
            elif 'xlsx' in filename:

                df = pd.read_excel(io.BytesIO(decoded) )

        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        unNamedCols = df.filter(regex='Unnamed').columns

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

    return df_uploaded.to_dict('records'), [{"name": i, "id": i} for i in df_uploaded.columns]


# update x-dropdown
@app.callback(Output('dropdown_target', 'options'),
              [Input('datatable-upload', 'contents'),
               Input('datatable-upload', 'filename'),
                   Input("data_selection_radio", "value"),])

def update_target_dropdown(contents, filename, data_selection):
    if contents is not None:
        df1 = parse_contents(contents, filename)
        df1 = df1.convert_dtypes()
        numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
        str_features = df1.select_dtypes(include=[ 'object','string']).columns

        if data_selection == 'None':
            df1 = df1
        elif data_selection == 'Categorical' and len(str_features) > 0:
            df1 = df1[[str_features]]
        elif data_selection == 'Numerical'  and len(numeric_features) > 0:
            df1 = df1[[numeric_features]]
        else:
            df1 = df1

        df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                    xaxis_column_name, yaxis_column_name,
                    xaxis_type, yaxis_type, data_selection, datasize)

        columns = columnsN.tolist()

        if df1 is not None:
            return [ {'label': x, 'value': x} for x in columns]
        else:
            return []
    else:
        return []

# update y-dropdown
@app.callback(Output('crossfilter-yaxis-column', 'options'),
              [Input('datatable-upload', 'contents'),
                Input('datatable-upload', 'filename'),
                Input("data_selection_radio", "value"),])

def update_y_dropdown(contents, filename, data_selection ):
    if contents is not None:
        df1 = parse_contents(contents, filename)
        df1 = df1.convert_dtypes()
        numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
        str_features = df1.select_dtypes(include=[ 'object','string']).columns


        if data_selection == 'None':
            df1 = df1
        elif data_selection == 'Categorical' and len(str_features) > 0:
            df1 = df1[[str_features]]
        elif data_selection == 'Numerical'  and len(numeric_features) > 0:
            df1 = df1[[numeric_features]]
        else:
            df1 = df1

        df1, str_features, numeric_features,   columnsN  = data_processing(df1, target, hoverData,
                    xaxis_column_name, yaxis_column_name,
                    xaxis_type, yaxis_type, data_selection, datasize)

        columns =  columnsN.tolist()

        if df1 is not None:
            return [ {'label': x, 'value': x} for x in columns]
        else:
            return []
    else:
        return []


# update x-dropdown
@app.callback(Output('crossfilter-xaxis-column', 'options'),
              [Input('datatable-upload', 'contents'),
               Input('datatable-upload', 'filename'),
                   Input("data_selection_radio", "value"),])

def update_x_dropdown(contents, filename, data_selection):

    if contents is not None:

        df1 = parse_contents(contents, filename)
        df1 = df1.convert_dtypes()
        numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
        str_features = df1.select_dtypes(include=[ 'object','string']).columns


        if data_selection == 'None':
            df1 = df1
        elif data_selection == 'Categorical' and len(str_features) > 0:
            df1 = df1[[str_features]]
        elif data_selection == 'Numerical'  and len(numeric_features) > 0:
            df1 = df1[[numeric_features]]
        else:
            df1 = df1

        df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                    xaxis_column_name, yaxis_column_name,
                    xaxis_type, yaxis_type, data_selection, datasize)

        columns =  columnsN.tolist()

        if df1 is not None:
            return [ {'label': x, 'value': x} for x in columns]
        else:
            return []
    else:
        return []

###############################################################################
#                crossfilter-indicator-scatter
###############################################################################

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
    prevent_initial_call=True)

def update_scat(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):


    df1 = pd.DataFrame(rows)

    if (df1.empty or len(df1.columns) < 1):
        fig = {'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
        return fig
    else:
        if target:
            df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                        xaxis_column_name, yaxis_column_name,
                        xaxis_type, yaxis_type, data_selection, datasize)

            df1.columns = columnsN

            df1 = df1.convert_dtypes()
            numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
            str_features = df1.select_dtypes(include=[ 'object','string']).columns

            if data_selection == 'None':
                df1 = df1
            elif data_selection == 'Categorical' and len(str_features) > 0:
                df1 = df1[[str_features]]
            elif data_selection == 'Numerical'  and len(numeric_features) > 0:
                df1 = df1[[numeric_features]]
            else:
                df1 = df1

        title = ( str(target) + ' vs '+ str(xaxis_column_name))

    return create_scatter(df1, xaxis_type, yaxis_type, xaxis_column_name, yaxis_column_name, title, target)#fig

#################################################################################

def create_scatter(dff, xaxis_type, yaxis_type, xaxis_column_name, yaxis_column_name, title, target):


    fig = px.scatter(dff, x=xaxis_column_name, y=target, title=title)

    fig.update_xaxes(showgrid=False)

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)' )

    fig.update_traces(mode='markers', marker_line_width=[3]*len(dff),
                      marker_size=[20]*len(dff) )

    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    fig.update_layout(margin={'l': 50, 'b': 50, 't': 50, 'r': 50}, hovermode='closest')


    fig.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="red",
        title_font_size =15,
        legend_title_font_color="black"
    )


    fig.update_layout(
        title={
            'y':1 ,
            'x':0.5,
            'font_size':20,
            'xanchor': 'center',
            'yanchor': 'top'})

    return fig
#################################################################################

def create_time_series(dff, xaxis_type, yaxis_type, ydata, title, target):

    fig = px.scatter(dff, x=ydata, y=target, title=title)
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(type='linear' if xaxis_type == 'Linear' else 'log')
    fig.update_yaxes(type='linear' if yaxis_type == 'Linear' else 'log')
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)' )

    fig.update_layout(height=225, margin={'l': 30, 'b': 30, 'r': 30, 't': 30})


    fig.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="red",
        title_font_size =15,
        legend_title_font_color="black"
    )

    fig.update_layout(
        title={
            'y':1 ,
            'x':0.5,
            'font_size':20,
            'xanchor': 'center',
            'yanchor': 'top'})

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
    prevent_initial_call=True)

def update_y_timeseries(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):

    df1 = pd.DataFrame(rows)

    if (df1.empty or len(df1.columns) < 1):
        fig = {'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
        return fig
    else:
        if target:
            df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                        xaxis_column_name, yaxis_column_name,
                        xaxis_type, yaxis_type, data_selection, datasize)

            df1.columns = columnsN

            df1 = df1.convert_dtypes()
            numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
            str_features = df1.select_dtypes(include=[ 'object','string']).columns

            if data_selection == 'None':
                df1 = df1
            elif data_selection == 'Categorical' and len(str_features) > 0:
                df1 = df1[[str_features]]
            elif data_selection == 'Numerical'  and len(numeric_features) > 0:
                df1 = df1[[numeric_features]]
            else:
                df1 = df1

        title = ( str(target) + ' vs '+ str(yaxis_column_name))

        return create_time_series(df1,  xaxis_type, yaxis_type, yaxis_column_name, title, target)

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
    if (df1.empty or len(df1.columns) < 1):
        fig = {'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
        return fig
    else:
        if target:
            df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                        xaxis_column_name, yaxis_column_name,
                        xaxis_type, yaxis_type, data_selection, datasize)

            df1.columns = columnsN

            df1 = df1.convert_dtypes()
            numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
            str_features = df1.select_dtypes(include=[ 'object','string']).columns


            if data_selection == 'None':
                df1 = df1
            elif data_selection == 'Categorical' and len(str_features) > 0:
                df1 = df1[[str_features]]
            elif data_selection == 'Numerical'  and len(numeric_features) > 0:
                df1 = df1[[numeric_features]]
            else:
                df1 = df1

        title = ( str(target) + ' vs '+ str(xaxis_column_name))
        return create_time_series(df1, xaxis_type, yaxis_type, xaxis_column_name, title, target)

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
    if (df1.empty or len(df1.columns) < 1):
        fig = {'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
    else:
        if target:
            df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                        xaxis_column_name, yaxis_column_name,
                        xaxis_type, yaxis_type, data_selection, datasize)


            df1.columns = columnsN

            df1 = df1.convert_dtypes()
            numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
            str_features = df1.select_dtypes(include=[ 'object','string']).columns


            if data_selection == 'None':
                df1 = df1
            elif data_selection == 'Categorical' and len(str_features) > 0:
                df1 = df1[[str_features]]
            elif data_selection == 'Numerical'  and len(numeric_features) > 0:
                df1 = df1[[numeric_features]]
            else:
                df1 = df1
            # try:
            title = ( str(target) + ' and its correlation with other vars')


            corr, colls =  do_corr(df1, imp_col=target)
            fig = px.imshow(corr)

            fig.update_layout(title=title, yaxis_zeroline=False, xaxis_zeroline=False)
            fig.update_layout(
                font_family="Courier New",
                font_color="black",
                title_font_family="Times New Roman",
                title_font_color="red",
                title_font_size =15,
                legend_title_font_color="black"
            )

            fig.update_layout(
                title={
                    'y':1 ,
                    'x':0.5,
                    'font_size':20,
                    'xanchor': 'center',
                    'yanchor': 'top'})
        else:
                fig = {'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }

    return fig

#################################################################################
#                correlationPlot
#################################################################################

@app.callback(
    Output('corr_network', 'figure'),
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
def update_corr_nx(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)
    if (df1.empty or len(df1.columns) < 1):
        fig = {'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
    else:
        if target:
            df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                        xaxis_column_name, yaxis_column_name,
                        xaxis_type, yaxis_type, data_selection, datasize)

            df1.columns = columnsN

            df1 = df1.convert_dtypes()
            numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
            str_features = df1.select_dtypes(include=[ 'object','string']).columns


            if data_selection == 'None':
                df1 = df1
            elif data_selection == 'Categorical' and len(str_features) > 0:
                df1 = df1[[str_features]]
            elif data_selection == 'Numerical'  and len(numeric_features) > 0:
                df1 = df1[[numeric_features]]
            else:
                df1 = df1
            # try:
            title = ( str(target) + ' and its correlation with other vars, network')

            corr, colls =  do_corr(df1, imp_col=target)
            links = corr.stack().reset_index()
            links.columns = ['var1', 'var2', 'value']
            links_filtered = links.loc[ (links['value'] > 0 ) & (links['var1'] != links['var2']) ]

            # Build your graph
            G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')

            # Plot the network:
            fig = nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black',
                    linewidths=1, font_size=15)

            fig.update_layout(title=title, yaxis_zeroline=False, xaxis_zeroline=False)
            fig.update_layout(
                font_family="Courier New",
                font_color="black",
                title_font_family="Times New Roman",
                title_font_color="red",
                title_font_size =15,
                legend_title_font_color="black"
            )


            fig.update_layout(
                title={
                    'y':1 ,
                    'x':0.5,
                    'font_size':20,
                    'xanchor': 'center',
                    'yanchor': 'top'})
        else:
                fig = {'data': [{
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

            title = ('Variable and their distributions'  )

            fig.update_layout(title=title, yaxis_zeroline=False, xaxis_zeroline=False),

            fig.update_layout(margin={'l': 50, 'b': 50, 't': 50, 'r': 50})

            fig.update_layout(
                font_family="Courier New",
                font_color="black",
                title_font_family="Times New Roman",
                title_font_color="red",
                title_font_size =20,
                legend_title_font_color="Black"
            )

            fig.update_layout(
                title={
                    'y':1,
                    'x':0.5,
                    'font_size':20,
                    'xanchor': 'center',
                    'yanchor': 'top'})

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
    prevent_initial_call=True)

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

#################################################################################
#               variable_importance_bar generate df
#################################################################################
@app.callback(
    dash.dependencies.Output('variable_importance_bar', 'figure'),
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
#################################################################################
#               variable_importance_bar generate bar fig
#################################################################################
def variable_importance_generator(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)
    if (df1.empty or len(df1.columns) < 1):
        fig = {
        'data': [{
            'x': [],
            'y': [],
            'type': 'scatter'
            }]
        }

        return  fig
    else:
        if target:
            df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                        xaxis_column_name, yaxis_column_name,
                        xaxis_type, yaxis_type, data_selection, datasize)

            df1.columns = columnsN

            df1 = df1.convert_dtypes()
            numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
            str_features = df1.select_dtypes(include=[ 'object','string']).columns

            if data_selection == 'None':
                df1 = df1
            elif data_selection == 'Categorical' and len(str_features) > 0:
                df1 = df1[[str_features]]
            elif data_selection == 'Numerical'  and len(numeric_features) > 0:
                df1 = df1[[numeric_features]]
            else:
                df1 = df1

            df1, y, columnsN  = data_transformation(df1, target)

            rf , y_pred_all, y_pred_best, y_test, fi, impvar, perf = RF_predict(df1 , y)

            title = ('RF var importance for target var:' + str(target))

            fig = go.Figure(go.Bar(
                        x=fi.importance,
                        y=fi.feature,
                        text=fi.importance,
                        textposition='auto',
                        orientation='h',
                        ))

            fig.update_layout(title=title, yaxis_zeroline=False, xaxis_zeroline=False)

            fig.update_layout(margin={'l': 50, 'b': 50, 't': 50, 'r': 50})

            fig.update_layout(
                font_family="Courier New",
                font_color="black",
                title_font_family="Times New Roman",
                title_font_color="red",
                title_font_size =20,
                legend_title_font_color="Black"
            )

            fig.update_layout(
                title={
                    'y':1,
                    'x':0.5,
                    'font_size':20,
                    'xanchor': 'center',
                    'yanchor': 'top'})
        else:
            fig = {
            'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
        return fig
    return fig

def plot_wordcloud(data):
    d = {a: x for a, x in data.values}
    wc = WordCloud(background_color='black' )
    wc.fit_words(d)
    return wc.to_image()
#################################################################################
#               variable_importance_bar generate wordcloud
#################################################################################

@app.callback(
    dash.dependencies.Output('variable_importance_wordcloud', 'figure'),

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
def generate_variable_importance_wordcloud_graph(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):

    df1 = pd.DataFrame(rows)
    if target:
        df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                    xaxis_column_name, yaxis_column_name,
                    xaxis_type, yaxis_type, data_selection, datasize)

        df1.columns = columnsN

        df1 = df1.convert_dtypes()
        numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
        str_features = df1.select_dtypes(include=[ 'object','string']).columns


        if data_selection == 'None':
            df1 = df1
        elif data_selection == 'Categorical' and len(str_features) > 0:
            df1 = df1[[str_features]]
        elif data_selection == 'Numerical'  and len(numeric_features) > 0:
            df1 = df1[[numeric_features]]
        else:
            df1 = df1

        df1, y,  columnsN  = data_transformation(df1, target)

        rf , y_pred_all, y_pred_best, y_test, fi, impvar, perf = RF_predict(df1 , y)
        x = rf.feature_importances_

        weights = fi.importance.values
        weights = (1 + weights / weights.max() * 50)
        words = fi.feature

        colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(len(words))]

        title = ('RF var importance wordcloud for target var:' + str(target))

        data = go.Scatter(x = list(range(  -round(len(x)/2), round(len(words)/2)+1,  1)),
                          y= x,
                          mode='text',
                          text=words,
                          marker={'opacity': 0.3},
                          textfont={'size':  weights, #scale weights to 1 -10
                                    'color': colors})
        layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})
        fig = go.Figure(data=[data], layout=layout)

        fig.update_layout(title=title, yaxis_zeroline=False, xaxis_zeroline=False),

        fig['layout']['yaxis'].update( range=[np.min(x)-5, np.max(x)+5],  autorange=False)
        fig['layout']['xaxis'].update(  range=[np.min(x)-5, np.max(x)+5], autorange=False)

        fig.update_layout(margin={'l': 50, 'b': 50, 't': 50, 'r': 50})


        fig.update_layout(
            font_family="Courier New",
            font_color="black",
            title_font_family="Times New Roman",
            title_font_color="red",
            title_font_size =20,
            legend_title_font_color="Black"
        )

        fig.update_layout(
            title={
                'y':1,
                'x':0.5,
                'font_size':20,
                'xanchor': 'center',
                'yanchor': 'top'})

        return fig

    else:
        fig = {'data': [{
                        'x': [],
                        'y': [],
                        'type': 'scatter'
                    }]
                }
    return fig

@app.callback(
    dash.dependencies.Output('RF_plot', 'figure'),

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
def rf_graph(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):

    df1 = pd.DataFrame(rows)
    if target:
        df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                    xaxis_column_name, yaxis_column_name,
                    xaxis_type, yaxis_type, data_selection, datasize)

        df1.columns = columnsN

        df1 = df1.convert_dtypes()
        numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
        str_features = df1.select_dtypes(include=[ 'object','string']).columns

        if data_selection == 'None':
            df1 = df1
        elif data_selection == 'Categorical' and len(str_features) > 0:
            df1 = df1[[str_features]]
        elif data_selection == 'Numerical'  and len(numeric_features) > 0:
            df1 = df1[[numeric_features]]
        else:
            df1 = df1

        df1, y,  columnsN  = data_transformation(df1, target)

        rf , y_pred_all, y_pred_best, y_test, fi, impvar, perf = RF_predict(df1 , y)
        fig = tree.plot_tree(rf)

        title = ('RF var importance wordcloud for target var:' + str(target))

        fig.update_layout(title=title, yaxis_zeroline=False, xaxis_zeroline=False),

        fig.update_layout(margin={'l': 50, 'b': 50, 't': 50, 'r': 50})

        fig.update_layout(
            font_family="Courier New",
            font_color="black",
            title_font_family="Times New Roman",
            title_font_color="red",
            title_font_size =20,
            legend_title_font_color="Black"
        )

        fig.update_layout(
            title={
                'y':1,
                'x':0.5,
                'font_size':20,
                'xanchor': 'center',
                'yanchor': 'top'})

        return fig

    else:
        fig = {'data': [{
                        'x': [],
                        'y': [],
                        'type': 'scatter'
                    }]
                }
    return fig


#################################################################################
#               model performance
#################################################################################
@app.callback(
    dash.dependencies.Output('performance_whisker', 'figure'),
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

def update_pairplot_whisker1(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)
    if (df1.empty or len(df1.columns) < 1):
        fig = {
        'data': [{
            'x': [],
            'y': [],
            'type': 'scatter'
            }]
        }

        return  fig
    else:
        if target:
            df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                        xaxis_column_name, yaxis_column_name,
                        xaxis_type, yaxis_type, data_selection, datasize)

            df1.columns = columnsN

            df1 = df1.convert_dtypes()
            numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
            str_features = df1.select_dtypes(include=[ 'object','string']).columns

            if data_selection == 'None':
                df1 = df1
            elif data_selection == 'Categorical' and len(str_features) > 0:
                df1 = df1[[str_features]]
            elif data_selection == 'Numerical'  and len(numeric_features) > 0:
                df1 = df1[[numeric_features]]
            else:
                df1 = df1

            df1, y, columnsN  = data_transformation(df1, target)

            rf , y_pred_all, y_pred_best, y_test, fi, impvar, perf = RF_predict(df1 , y)

            np.random.seed(1)

            N = len(y_test)
            xss = list(range(0,  N))
            title = ('RF predictions for target var:' + str(target))
            layout = go.Layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    x=0.2,
                    y=0.9,
                    traceorder='normal',
                    font=dict(
                        size=12,),
                ),
                annotations=[
                    dict(
                        x=0,
                        y=0.85,
                        xref='paper',
                        yref='paper',
                        showarrow=False
                    )
                ]
            )

            fig = go.Figure(layout = layout)

            nm  = ('target var: ' + str(target))
            bestPrdName = ('preduction from 1st imp var: ' + str(impvar))
            allPredName = ('prediction from all vars' )

            fig.add_trace(go.Scatter(x=xss, y=y_test, name=nm,
                                     line=dict(color='royalblue', width=4)))


            fig.add_trace(go.Scatter(x=xss, y=y_pred_best, name= bestPrdName,
                                     line=dict(color='firebrick', width=4,
                                          dash='dash')))

            fig.add_trace(go.Scatter(x=xss, y=y_pred_all, name= allPredName,
                                     line = dict(color='firebrick', width=4, dash='dot')))

            fig.update_xaxes(showgrid=False)

            fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                               xref='paper', yref='paper', showarrow=False, align='left',
                               bgcolor='rgba(255, 255, 255, 0.5)' )

            fig.update_layout(margin={'l': 50, 'b': 50, 't': 50, 'r': 50}, hovermode='closest')


            fig.update_layout(
                font_family="Courier New",
                font_color="black",
                title_font_family="Times New Roman",
                title_font_color="Black",
                title_font_size =15,
                legend_title_font_color="black"
            )

            fig.update_layout(
                title={
                    'y':1 ,
                    'x':0.5,
                    'font_size':20,
                    'xanchor': 'center',
                    'yanchor': 'top'})


            fig.update_layout(title=title,
                   xaxis_title='values',
                   yaxis_title=target)
        else:
            fig = {
            'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
        return fig
    return fig

#################################################################################
#              model accuracies
#################################################################################
@app.callback(
    dash.dependencies.Output('roc', 'figure'),
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
def roc_plot(rows, target, hoverData, xaxis_column_name, yaxis_column_name,
                xaxis_type, yaxis_type, data_selection, datasize):
    df1 = pd.DataFrame(rows)
    if (df1.empty or len(df1.columns) < 1):
        fig = {
        'data': [{
            'x': [],
            'y': [],
            'type': 'scatter'
            }]
        }

        return  fig
    else:
        if target:
            df1, str_features, numeric_features,  columnsN  = data_processing(df1, target, hoverData,
                        xaxis_column_name, yaxis_column_name,
                        xaxis_type, yaxis_type, data_selection, datasize)

            df1.columns = columnsN

            df1 = df1.convert_dtypes()
            numeric_features = df1.select_dtypes(include=['int64', 'float64','int32', 'float32' ]).columns
            str_features = df1.select_dtypes(include=[ 'object','string']).columns

            if data_selection == 'None':
                df1 = df1
            elif data_selection == 'Categorical' and len(str_features) > 0:
                df1 = df1[[str_features]]
            elif data_selection == 'Numerical'  and len(numeric_features) > 0:
                df1 = df1[[numeric_features]]
            else:
                df1 = df1

            df1, y, columnsN  = data_transformation(df1, target)

            rf , y_pred_all, y_pred_best, y_test, fi, impvar, perf = RF_predict(df1 , y)

            np.random.seed(1)

            N = len(perf)
            xss = list(range(0,  N)) #np.linspace(0, N, N)

            title = ('RF pred. accuracies for target var:' + str(target))

            layout = go.Layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    x=0.2,
                    y=0.85,
                    traceorder='normal',
                    font=dict(
                        size=12,),
                ),
                annotations=[
                    dict(
                        x=0,
                        y=0.75,
                        xref='paper',
                        yref='paper',
                        showarrow=False
                    )
                ]
            )

            fig = go.Figure(layout = layout)


            bestPrdName = ('preduction from 1st imp var: ' + str(impvar))
            allPredName = ('prediction from all vars')


            fig.add_trace(go.Scatter(x=xss, y=perf['Most important predector'], name= bestPrdName,
                                     line=dict(color='firebrick', width=4,
                                          dash='dash')))

            fig.add_trace(go.Scatter(x=xss, y=perf['All features'], name= allPredName,
                                     line = dict(color='firebrick', width=4, dash='dot')))

            fig.update_xaxes(showgrid=False)

            fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                               xref='paper', yref='paper', showarrow=False, align='left',
                               bgcolor='rgba(255, 255, 255, 0.5)' )

            fig.update_layout(margin={'l': 50, 'b': 50, 't': 50, 'r': 50}, hovermode='closest')


            fig.update_layout(
                font_family="Courier New",
                font_color="black",
                title_font_family="Times New Roman",
                title_font_color="Black",
                title_font_size =15,
                legend_title_font_color="black"
            )


            fig.update_layout(
                title={
                    'y':1 ,
                    'x':0.5,
                    'font_size':20,
                    'xanchor': 'center',
                    'yanchor': 'top'})


            fig.update_layout(title=title,
                   xaxis_title='values',
                   yaxis_title='Accuracy measures (1 = 100%)')

            fig.update_layout(
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = xss,
                    ticktext = perf.index
                )
            )

        else:
            fig = {
            'data': [{
                'x': [],
                'y': [],
                'type': 'scatter'
            }]
        }
        return fig
    return fig


if __name__ == '__main__':
    app.run_server()

