# Necessary Imports
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.linear_model import LinearRegression
import sankey_builder_2 as sk
from data_man import df_raw, country_data, encoded_image, columns, data


def main(): # Defines app
    # step 1: define the app object
    app = Dash(__name__)
    # step 2: define the layout
    # Defines the layout
    app.layout = html.Div(
        children=[
            html.H1('Myocardial Infarction Data Dashboard'),
            # First Column
            html.Div(
                children=[

                    html.H2('Data Sankey'),
                    dcc.Graph(id='sankey'),
                    html.P("Columns"),
                    dcc.RangeSlider(
                        id='data_columns',
                        min=0,
                        max=(len(columns) - 10),
                        marks={i: columns[i] for i in range(0, len(columns) - 9)},
                        value=[0, 2],
                        step=1,
                        allowCross=False
                    ),
                    html.P("Filter"),
                    dcc.Slider(id='cull', min=1, max=500, step=50, value=100),
                    html.H2("Feature Importance RandomForestRegressor"),
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width':'90%', 'height': '70%'})
                ],
                style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top', 'float': 'left'}
            ),

            # Second Column
            html.Div(
                children=[
                    html.H2(''),
                    html.H2("Map of Where Data was Sampled From"),
                    dcc.Graph(
                        id='world-map',
                        figure=px.choropleth(
                            country_data,
                            locations='Country',
                            locationmode='country names',
                            color='Count'
                        )
                    ),
                    html.Div(id='output-container'),
                    html.H2("Linear Regression"),
                    dcc.Graph(id="graph"),
                    html.P("Selection One"),
                    dcc.RadioItems(
                        id='radio',
                        options=['Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History',
                                 'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
                                 'Medication Use', 'Stress Level',
                                 'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
                                 'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Heart Attack Risk'],
                        value='Age',
                        inputStyle={"margin-left": "20px"}
                    ),
                    html.P("Selection Two"),
                    dcc.Dropdown(
                        id='dropdown2',
                        options=['Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History',
                                 'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week',
                                 'Diet', 'Medication Use', 'Stress Level',
                                 'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
                                 'Physical Activity Days Per Week', 'Sleep Hours Per Day',
                                 'Heart Attack Risk'],
                        value='Sex'
                    ),

                ],
                style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top', 'float': 'right'}
            ),
        ]
    )
    columns_list = df_raw.columns.tolist()

    @app.callback(
        Output('sankey', 'figure'),
        Input('cull', 'value'),
        Input('data_columns', 'value'),

    )
    def update_sankey(cull, data_columns):
        try:
            in_values = columns[data_columns[0]:data_columns[1] + 1]
            title = f'Graph of Factors {in_values[0]} Through {in_values[-1]}'
            fig = sk.make_sankey(data, list(in_values), title, cull=True, cull_val=cull)
            return fig
        except:
            fig = {}
            return fig

    @app.callback(
        Output('output-container', 'children'),
        Output('graph', 'figure'),
        Input('radio', 'value'),
        Input('dropdown2', 'value')

    )
    def update_LinearRegressor_and_input(value1, value2): # Builds Linear regression graph
        # Your first callback logic here
        output_text = ' '
        if str(value1) in columns_list:
            column1 = f'{value1}'
            column2 = f'{value2}'
            x = df_raw[column1]
            y = df_raw[column2]

            # Create and fit the linear regression model
            model = LinearRegression()
            model.fit(x.values.reshape(-1, 1), y)

            # Generate predictions using the model
            x_pred = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            y_pred = model.predict(x_pred)

            # Create a scatter plot of observed data and a line plot of predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=str(column1) + ' and ' + str(column2), marker=dict(color='black')))
            fig.add_trace(
                go.Scatter(x=x_pred.flatten(), y=y_pred, mode='lines', line=dict(color='red'), name='Correlation'))
            fig.update_layout(
                width=800,
                height=400,
                xaxis_title=str(column1),
                yaxis_title=str(column2),
                legend=dict(x=0, y=1)
            )

            return output_text, fig

    # step 4: Run the server
    app.run_server(debug=True)


main() # Calls app
