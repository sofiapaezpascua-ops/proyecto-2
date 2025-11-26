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
            htlm.Div([
                    htlm.div([id="price-info",
                              style={"fontSize": "18px","marginBottom": "1rem","padding": "1rem","backgroundColor": "white","borderRadius": "5px"
                    }),
                    dcc.Graph(
                        id="price-map",
                        style={"height": "70vh"})
                    ], style={ "width": "60%","padding": "2rem"})
                    ], style={"display": "flex", "gap": "2rem", "padding": "2rem"})
                        
@app.callback(
    [Output("price-map", "figure"),
     Output("price-info", "children")],
    Input("predict-button", "n_clicks"),
    [State("property-type", "value"),
     State("input-bathrooms", "value"),
     State("input-beds", "value"),
     State("input-rating", "value"),
     State("input-minimum_nights", "value"),
     State("input-availability_365d", "value"),
     State("input-host_acceptance_rate", "value"),
     State("amenities", "value")]
)
def generate_price_map(n_clicks, prop_type, bathrooms, beds, rating, 
                       min_nights, availability, host_rate, amenities):
    
    if n_clicks == 0:
        # Mapa inicial vacío
        fig = go.Figure()
        fig.update_layout(
            title="Presiona 'Generar Mapa' para ver predicciones",
            xaxis_title="Longitud",
            yaxis_title="Latitud",
            height=600
        )
        return fig, 
    
    # Crear grid de coordenadas (ajustado a las coordenadas de madrid)
    lat_min, lat_max = 40.35, 40.50
    lon_min, lon_max = -3.75, -3.60
    
    # Crear grid más denso para mejor visualización
    lats = np.linspace(lat_min, lat_max, 30)
    lons = np.linspace(lon_min, lon_max, 30)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Preparar características base
    wifi = 1 if "Wifi" in amenities else 0
    kitchen = 1 if "Kitchen_and_dining" in amenities else 0
    tv = 1 if "TV" in amenities else 0
    fridge = 1 if "Refrigerator" in amenities else 0
    essentials = 1 if "Essentials" in amenities else 0
    
    # Crear dataset para predicciones
    predicciones = []
    coords = []
    
    for lat in lats:
        for lon in lons:
            # Crear fila con todas las características
            fila = [
                host_rate,  # host_acceptance_rate
                lon,  # longitude
                lat,  # latitude
                bathrooms,  # bathrooms
                beds,  # beds
                min_nights,  # minimum_nights
                availability,  # availability_365d
                availability * 0.7,  # estimated_occupancy_365d (estimado)
                rating,  # review_scores_rating
                rating,  # review_scores_cleanliness
                rating,  # review_scores_communication
                rating,  # review_scores_value
                wifi,
                kitchen,
                tv,
                fridge,
                essentials,
                property_entire,
                property_other,
                property_shared
            ]
            
            coords.append((lat, lon))
            
            # PREDICCIÓN CON TU MODELO
            
            x_row = np.array([[fila[col] for col in feature_cols]], dtype="float32")
            precio_pred = float(model.predict(x_row, verbose=0)[0, 0])

            predicciones.append(max(precio_pred, 7))
    
    
    # Crear mapa con densidad de colores
    fig = px.density_mapbox(
        df_map,
        lat='latitude',
        lon='longitude',
        z='precio_pred',
        radius=20,
        center=dict(lat=(lat_min + lat_max)/2, lon=(lon_min + lon_max)/2),
        zoom=11,
        mapbox_style="open-street-map",
        color_continuous_scale="RdYlGn_r",  # Rojo (caro) a Verde (barato)
        title="Predicción de Precios por Ubicación",
        labels={'precio': 'Precio (USD/noche)'}
    )
    
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    
    # Información de resumen
    precio_min = df_map['precio_pred'].min()
    precio_max = df_map['precio_pred'].max()
    precio_prom = df_map['precio_pred'].mean()
    
    info = html.Div([
        html.H4("Resumen de Predicciones", style={"margin": "0 0 1rem 0"}),
        html.Div([
            html.Span("Precio Promedio: ", style={"fontWeight": "bold"}),
            html.Span(f"${precio_prom:.2f}/noche", style={"color": "#4CAF50"})
        ]),
        html.Div([
            html.Span("Precio Mínimo: ", style={"fontWeight": "bold"}),
            html.Span(f"${precio_min:.2f}/noche")
        ]),
        html.Div([
            html.Span("Precio Máximo: ", style={"fontWeight": "bold"}),
            html.Span(f"${precio_max:.2f}/noche")
        ])
    ])
    
    return fig, info

if __name__ == "__main__":
    app.run_server(debug=True)

        

    

  
