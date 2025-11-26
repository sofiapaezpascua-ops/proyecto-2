import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import tensorflow as tf

app = dash.Dash(__name__)

#Cargar datos

df = pd.read_csv('listings_limpio.csv') 

#Columnas a utilizar
columnas = [
    "host_acceptance_rate",
    "longitude",
    "latitude",
    "bathrooms",
    "beds",
    "minimum_nights",
    "availability_365",
    "estimated_occupancy_l365d",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_communication",
    "review_scores_value",
    "Wifi",
    "Kitchen_and_dining",
    "TV",
    "Refrigerator",
    "Essentials",
    "property_Entire_Place",
    "property_Other",
    "property_Shared_Room",
]
input = {"host_acceptance_rate": (0, 100, 0.1, "%"),
    "bathrooms": (0, 19, 1, ""),
    "beds": (0, 40, 1, ""),
    "minimum_nights": (1, 366, 1, " noches"),
    "availability_365": (1, 365, 1, " días"),
    "estimated_occupancy_l365d": (1, 255, 1, " días"),
    "review_scores_rating": (1, 5, 0.1, "/5"),
    "review_scores_cleanliness": (1, 5, 0.1, "/5"),
    "review_scores_communication": (1, 5, 0.1, "/5"),
    "review_scores_value": (1, 5, 0.1, "/5"),
    "Wifi": (0, 1, 1, ""),
    "Kitchen_and_dining": (0, 1, 1, ""),
    "TV": (0, 1, 1, ""),
    "Refrigerator": (0, 1, 1, ""),
    "Essentials": (0, 1, 1, ""),
    "property_Entire_Place": (0, 1, 1, ""),
    "property_Other": (0, 1, 1, ""),
    "property_Shared_Room": (0, 1, 1, ""),
}


modelo_regresion = tf.keras.models.load_model("red neuronal_airbnb.keras")

df["pred precio"]= modelo_regresion.predict(x)



#Layout del dash 

app.layout = html.Div(children=[ 
    html.H1(children="Predicción de Precios de Airbnb con Red Neuronal", 
            style={"textAlign": "center"}),
    #Inputs que describen el alojamiento
    htlm.Div([ htlm.H3("Seleccione las características del alojamiento:")]),
        html.Div([ 
                  htlm.Div([
                      htlm.Label("número de baños")
                      dcc.Slider(
                          id="bathrooms",
                          min=0,
                          max=19,
                          step=1,
                          value=1,
                          marks={i: str(i) for i in range(0, 20, 2)},
                      ),
                  ])
                  htlm.Div([
                      htlm.Label("número de camas")
                        dcc.Slider(
                            id="beds",
                            min=0,
                            max=40,
                            step=1,
                            value=1,
                            marks={i: str(i) for i in range(0, 41, 5)},
                        ),
                  ])
                    htlm.Div([
                        htlm.Label("mínimo de noches")
                            dcc.Slider(
                                id="minimum_nights",
                                min=1,
                                max=366,
                                step=1,
                                value=1,
                                marks={i: str(i) for i in range(1, 367, 30)},
                            ),
                            ])
                    htlm.Div([
                        htlm.Label("disponibilidad ")
                            dcc.Input(
                                id="availability_365",
                                type = "number",
                                min=1,
                                max=365,
                                value=300),
                            ])
                    htlm.Div([
                        htlm.Label("ocupación estimada ")
                            dcc.Input(
                                id="estimated_occupancy_l365d",
                                type = "number",
                                min=1,
                                max=255,
                                value=200),
                    ])
                    htlm.Div([
                        htlm.Label("puntuación de limpieza")
                            dcc.Slider(
                                id="review_scores_cleanliness",
                                min=1,
                                max=5,
                                step=0.1,
                                value=3,
                                marks={i: str(i) for i in range(1, 6)},
                            ),
                    ])
                    htlm.Div([
                        htlm.Label("puntuación de comunicación")
                            dcc.Slider(
                                id="review_scores_communication",
                                min=1,
                                max=5,
                                step=0.1,
                                value=3,
                                marks={i: str(i) for i in range(1, 6)},
                            ),
                    ])
                    htlm.Div([
                        htlm.Label("puntuación de valor")
                            dcc.Slider(
                                id="review_scores_value",
                                min=1,
                                max=5,
                                step=0.1,
                                value=3,
                                marks={i: str(i) for i in range(1, 6)},
                            ),
                    ])
                    htlm.Div([
                        htlm.Label("Seleccione las ameniades:")
                        dcc.Checklist(
                            id="amenities",
                            options=[
                                {"label": "Wifi", "value": "Wifi"},
                                {"label": "Kitchen_and_dining", "value": "Kitchen_and_dining"},
                                {"label": "TV", "value": "TV"},
                                {"label": "Refrigerator", "value": "Refrigerator"},
                                {"label": "Essentials", "value": "Essentials"},
                            ],
                            value=["Wifi", "Kitchen_and_dining", "TV", "Refrigerator", "Essentials"],
                        ),
                    ])
                    htlm.Div([
                        htlm.Label("Tasa de aceptación del anfitrión (%)"), style={"fontWeight": "bold"}),
                        dcc.Input(id="input-host_acceptance_rate", 
                                  type="number", 
                                  value=90, 
                                  min=0, 
                                  max=100,
                             style={"width": "100%", "padding": "0.5rem"}
                    ])
                    htlm.Button([ "Predecir Precio"
                                 id = "predict-button",
                                 n_clicks=0
                                 style={
                           "width": "100%",
                           "padding": "1rem",
                           "fontSize": "18px",
                           "backgroundColor": "#4CAF50",
                           "color": "white",
                           "border": "none",
                           "borderRadius": "5px",
                           "cursor": "pointer",
                           "marginTop": "1rem"
                       }]) 
            #Generar el mapa 
            htlm

])
    

  
