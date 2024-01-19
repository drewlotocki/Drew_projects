# Necessary Imports
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

def main():
    app = Dash(__name__)
    app.layout = html.Div(
        style={'backgroundColor': 'lightgrey'},  # Set the background color of the entire page
        children=[
            html.H1('Julia Graph Dashboard'),
            html.H2('', style={'color': 'black', 'font-family': 'Times New Roman'}),  # Styling for equation display
            dcc.Graph(id='julia_graph', figure={'layout': {'plot_bgcolor': 'lightgrey', 'paper_bgcolor': 'lightgrey'}}),
            html.P("Real Portion of Constant"),
            dcc.Slider(0, 5, 0.01, value=0, id='real_constant', marks=None,
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.P("Imaginary Portion of Constant"),
            dcc.Slider(0, 5, 0.01, value=0, id='imaginary_constant', marks=None,
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.P("Max Orbit Iterations"),
            dcc.Slider(0, 20, 1, value=10, id='max_iter', marks=None,
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]
    )

    @app.callback(
        [Output('julia_graph', 'figure'),
         Output('julia_graph', 'config')],
        [Input('real_constant', 'value'),
         Input('imaginary_constant', 'value'),
         Input('max_iter', 'value')]
    )
    def generate_julia_image(real, imaginary, max_iter):
        image = np.zeros((2500, 2500))

        for x in range(2500):
            for y in range(2500):
                zx = -2 + (x / (2500 - 1)) * (2 - -2)
                zy = -2 + (y / (2500 - 1)) * (2 - -2)

                value = julia_fractal(complex(zx, zy), complex(real, imaginary), max_iter)
                image[y, x] = value

        # Create the figure directly using Plotly's imshow
        fig = px.imshow(image, color_continuous_scale='Inferno')

        # Remove axis labels
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        # Display the equation in the title
        equation_text = f"Equation for Next Value in Orbit: f(z) = z\u00B2 + {real} + {imaginary}i"
        fig.update_layout(
            title_text=equation_text,
            title_font=dict(family="Times New Roman", size=18, color="black"),
            plot_bgcolor='lightgrey',  # Set the background color
            height=600,  # Set the height of the graph
            width=800   # Set the width of the graph
        )

        # Set up configuration for no modebar buttons
        config = {'modeBarButtonsToRemove': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']}

        return fig, config

    # Run the server
    app.run_server(debug=True)

def julia_fractal(z, c, max_iter):
    for i in range(max_iter):
        if abs(z) > 2.0:
            return i
        z = z ** 2 + c
    return max_iter

main()  # Calls app
