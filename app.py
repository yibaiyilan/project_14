import dash
from dash import dcc,html
from dash.dependencies import Input, Output, State
import pickle
import numpy as np

########### Define your variables ######
myheading1='Insurance Dataset'
tabtitle = 'Insurance'
sourceurl = 'https://github.com/yibaiyilan/project_13/blob/master/analysis/insurance.ipynb'
githublink = 'https://github.com/yibaiyilan/project_13'


########### open the pickle files ######
with open('analysis/model_components/coefs_fig.pkl', 'rb') as f:
    coefs=pickle.load(f)
with open('analysis/model_components/r2_fig.pkl', 'rb') as f:
    r2_fig=pickle.load(f)
with open('analysis/model_components/rmse_fig.pkl', 'rb') as f:
    rmse_fig=pickle.load(f)
with open('analysis/model_components/std_scaler.pkl', 'rb') as f:
    std_scaler=pickle.load(f)
with open('analysis/model_components/lin_reg.pkl', 'rb') as f:
    lin_reg=pickle.load(f)

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1('Insurance Charges'),
    html.H4('What are the insurance charges'),
    html.H6('Features of the Policy holder:'),

    ### Prediction Block
    html.Div(children=[

        html.Div([
                    html.Div('Age:'),
                    dcc.Input(id='age', value=40, type='number', min=18, max=64, step=1),

                    html.Div('Sex:'),
                    dcc.Input(id='sex_code', value=1, type='number', min=1, max=2, step=1),

                    html.Div('BMI'),
                    dcc.Input(id='bmi', value=30, type='number', min=16, max=53, step=.1),

                ], className='four columns'),

        html.Div([


                    html.Div('Children'),
                    dcc.Input(id='kid', value=1, type='number', min=0, max=5, step=1),

                    html.Div('Smoker'),
                    dcc.Input(id='smoker', value=1, type='number', min=1, max=2, step=1),

                    html.Div('Region'),
                    dcc.Input(id='region_code', value=1, type='number', min=1, max=4, step=1),

                ], className='four columns'),
        html.Div([
                    html.H6('Insurance Charges (Predicted):'),
                    html.Button(children='Submit', id='submit-val', n_clicks=0,
                                    style={
                                    'background-color': 'red',
                                    'color': 'white',
                                    'margin-left': '5px',
                                    'verticalAlign': 'center',
                                    'horizontalAlign': 'center'}
                                    ),

                    html.Div(id='Results')
                ], className='four columns')
            ], className='twelve columns'),
        ### Evaluation Block
        html.Div(children=[
            html.Div(
                    [dcc.Graph(figure=r2_fig, id='r2_fig')
                    ], className='six columns'),
            html.Div(
                    [dcc.Graph(figure=rmse_fig, id='rmse_fig')
                    ], className='six columns'),
                ], className='twelve columns'),

        html.Div(children=[
                html.H3('Linear Regression Coefficients (standardized features)'),
                dcc.Graph(figure=coefs, id='coefs_fig')
                ], className='twelve columns'),

        html.A('Code on Github', href=githublink),
        html.Br(),
        html.A("Data Source", href=sourceurl),
        ], className='twelve columns')


######### Define Callback
@app.callback(
    Output(component_id='Results', component_property='children'),
    Input(component_id='submit-val', component_property='n_clicks'),
    # regression inputs:
    State(component_id='age', component_property='value'),
    State(component_id='sex_code', component_property='value'),
    State(component_id='bmi', component_property='value'),
    State(component_id='kid', component_property='value'),
    State(component_id='smoker', component_property='value'),
    State(component_id='region_code', component_property='value'),
)
def make_prediction(clicks,age,sex_code,bmi,kid,smoker,region_code):
    if clicks==0:
        return "waiting for inputs"
    else:

        inputs=np.array([age,sex_code,bmi,kid,smoker,region_code]).reshape(1, -1)

        # standardization
        std_inputs = std_scaler.transform(inputs)

        y = lin_reg.predict(std_inputs)
        formatted_y = "${:,.2f}".format(y[0])
        return formatted_y



############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
