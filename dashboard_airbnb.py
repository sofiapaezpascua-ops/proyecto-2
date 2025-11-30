import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import warnings
warnings.filterwarnings('ignore')

app = dash.Dash(__name__)

# Cargar datos
df = pd.read_csv('listings_limpio.csv')

# Columnas a utilizar
columnas = [
    "host_acceptance_rate",
    "host_response_rate",
    "longitude",
    "latitude",
    "bathrooms",
    "beds",
    "bedrooms",
    "accommodates",
    "minimum_nights",
    "maximum_nights",
    "availability_365",
    "estimated_occupancy_l365d",
    "estimated_revenue_l365d",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_accuracy",
    "review_scores_checkin",
    "review_scores_communication",
    #"review_scores_location",
    "review_scores_value",
    "Wifi",
    "Kitchen_and_dining",
    "TV",
    "Refrigerator",
    "Essentials",
    "Air_conditioning",
    "Washer_dryer",
    "Safe",
    "Smoke_alarm_home_safety",
    "Services",
    "property_Entire_Place",
    "property_Other",
    "property_Shared_Room",
    "property_Hotel_Room",
    "property_Private_Room",
]


input_config = {
    "host_acceptance_rate": (0, 100, 0.1, "%"),
    "host_response_rate": (0, 100, 0.1, "%"),
    "bathrooms": (0, 19, 1, ""),
    "beds": (0, 40, 1, ""),
    "bedrooms": (0, 25, 1, ""),
    "accommodates": (1, 16, 1, " personas"),
    "minimum_nights": (1, 366, 1, " noches"),
    "maximum_nights": (1, 1125, 1, " noches"),
    "availability_365": (1, 365, 1, " días"),
    "estimated_occupancy_l365d": (1, 255, 1, " días"),
    "estimated_revenue_l365d": (7, 10000, 1, " USD"),
    "number_of_reviews": (0, 500, 1, ""),
    "reviews_per_month": (0, 10, 0.1, ""),
    "review_scores_rating": (1, 5, 0.1, "/5"),
    "review_scores_cleanliness": (1, 5, 0.1, "/5"),
    "review_scores_accuracy": (1, 5, 0.1, "/5"),
    "review_scores_checkin": (1, 5, 0.1, "/5"),
    "review_scores_communication": (1, 5, 0.1, "/5"),
    "review_scores_value": (1, 5, 0.1, "/5"),
    "Wifi": (0, 1, 1, ""),
    "Kitchen_and_dining": (0, 1, 1, ""),
    "TV": (0, 1, 1, ""),
    "Refrigerator": (0, 1, 1, ""),
    "Essentials": (0, 1, 1, ""),
    "Air_conditioning": (0, 1, 1, ""),
    "Washer_dryer": (0, 1, 1, ""),
    "Safe": (0, 1, 1, ""),
    "Smoke_alarm_home_safety": (0, 1, 1, ""),
    "Services": (0, 1, 1, ""),
    "property_Entire_Place": (0, 1, 1, ""),
    "property_Other": (0, 1, 1, ""),
    "property_Shared_Room": (0, 1, 1, ""),
    "property_Hotel_Room": (0, 1, 1, ""),
    "property_Private_Room": (0, 1, 1, ""),
}

#Traer la definicion de la funcion elu
def elu_plus_one(x):
    return tf.keras.activations.elu(x) + 1.0

# Cargar modelo regresion
modelo_regresion = tf.keras.models.load_model("modelo_airbnb.keras",custom_objects={"elu_plus_one": elu_plus_one},compile=False)

#Cargar modelo clasificacion
modelo_clasificacion = tf.keras.models.load_model("modelo_clasifacion_andes.keras",custom_objects={"elu_plus_one": elu_plus_one},compile=False)

#Definir la estructura del dash 
app.layout = html.Div(
    style={
        "backgroundColor": "#f5f7fa",
        "minHeight": "100vh",
        "padding": "30px",
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
    children=[
        html.Div(
            style={
                "maxWidth": "1200px",
                "margin": "0 auto",
                "backgroundColor": "white",
                "borderRadius": "16px",
                "boxShadow": "0 4px 18px rgba(0,0,0,0.08)",
                "padding": "24px 28px",
                "display": "flex",
                "flexDirection": "row",
                "gap": "32px",
            },
            children=[
                # --------- COLUMNA IZQUIERDA (CONTROLES) ----------
                html.Div(
                    style={
                        "width": "35%",
                        "maxWidth": "420px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "18px",
                        "maxHeight": "80vh",
                        "overflowY": "auto",
                        "paddingRight": "6px",
                    },
                    children=[
                        html.H2(
                            "Predicción de precios Airbnb",
                            style={
                                "textAlign": "left",
                                "marginBottom": "4px",
                            },
                        ),
                        html.P(
                            "Ajusta las características del alojamiento y genera el mapa de precios estimados.",
                            style={
                                "color": "#555",
                                "fontSize": "13px",
                                "marginBottom": "10px",
                            },
                        ),

                        html.H4("1. Tipo de alojamiento", style={"marginTop": "10px"}),
                        html.Div(
                            style={
                                "border": "1px solid #e3e6ee",
                                "borderRadius": "10px",
                                "padding": "10px 12px",
                            },
                            children=[
                                html.Label(
                                    "Tipo de propiedad",
                                    style={"fontWeight": "600", "fontSize": "13px"},
                                ),
                                dcc.Dropdown(
                                    id="property-type",
                                    options=[
                                        {"label": "Lugar completo", "value": "entire"},
                                        {"label": "Habitación privada", "value": "private"},
                                        {"label": "Habitación compartida", "value": "shared"},
                                        {"label": "Hotel", "value": "hotel"},
                                        {"label": "Otro", "value": "other"},
                                    ],
                                    value="entire",
                                    clearable=False,
                                ),
                            ],
                        ),

                        html.H4("2. Capacidad y distribución"),
                        html.Div(
                            style={
                                "border": "1px solid #e3e6ee",
                                "borderRadius": "10px",
                                "padding": "10px 12px",
                            },
                            children=[
                                html.Label("Número de baños"),
                                dcc.Slider(
                                    id="input-bathrooms",
                                    min=0,
                                    max=5,
                                    step=0.5,
                                    value=1,
                                    marks={i: str(i) for i in range(0, 6)},
                                ),
                                html.Br(),
                                html.Label("Número de camas"),
                                dcc.Slider(
                                    id="input-beds",
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=2,
                                    marks={i: str(i) for i in range(1, 11)},
                                ),
                                html.Br(),
                                html.Label("Número de habitaciones"),
                                dcc.Slider(
                                    id="input-bedrooms",
                                    min=0,
                                    max=10,
                                    step=1,
                                    value=1,
                                    marks={i: str(i) for i in range(0, 11)},
                                ),
                                html.Br(),
                                html.Label("Capacidad de huéspedes"),
                                dcc.Slider(
                                    id="input-accommodates",
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=2,
                                    marks={i: str(i) for i in range(1, 11)},
                                ),
                            ],
                        ),

                        html.H4("3. Estancia y ocupación"),
                        html.Div(
                            style={
                                "border": "1px solid #e3e6ee",
                                "borderRadius": "10px",
                                "padding": "10px 12px",
                            },
                            children=[
                                html.Label("Mínimo de noches"),
                                dcc.Input(
                                    id="input-minimum_nights",
                                    type="number",
                                    min=1,
                                    max=365,
                                    value=2,
                                    style={"width": "100%"},
                                ),
                                html.Br(), html.Br(),
                                html.Label("Máximo de noches"),
                                dcc.Input(
                                    id="input-maximum_nights",
                                    type="number",
                                    min=1,
                                    max=365,
                                    value=30,
                                    style={"width": "100%"},
                                ),
                                html.Br(), html.Br(),
                                html.Label("Disponibilidad (días al año)"),
                                dcc.Input(
                                    id="input-availability_365d",
                                    type="number",
                                    min=0,
                                    max=365,
                                    value=200,
                                    style={"width": "100%"},
                                ),
                            ],
                        ),

                        html.H4("4. Reseñas y puntuaciones"),
                        html.Div(
                            style={
                                "border": "1px solid #e3e6ee",
                                "borderRadius": "10px",
                                "padding": "10px 12px",
                            },
                            children=[
                                html.Label("Puntuación general"),
                                dcc.Slider(
                                    id="input-rating",
                                    min=1,
                                    max=5,
                                    step=0.1,
                                    value=4.5,
                                    marks={i: str(i) for i in range(1, 6)},
                                ),
                                html.Br(),
                                html.Label("Puntuación de limpieza"),
                                dcc.Slider(
                                    id="input-review_scores_cleanliness",
                                    min=1,
                                    max=5,
                                    step=0.1,
                                    value=4.5,
                                    marks={i: str(i) for i in range(1, 6)},
                                ),
                                html.Br(),
                                html.Label("Exactitud"),
                                dcc.Slider(
                                    id="input-review_scores_accuracy",
                                    min=1,
                                    max=5,
                                    step=0.1,
                                    value=4.5,
                                    marks={i: str(i) for i in range(1, 6)},
                                ),
                                html.Br(),
                                html.Label("Check-in"),
                                dcc.Slider(
                                    id="input-review_scores_checkin",
                                    min=1,
                                    max=5,
                                    step=0.1,
                                    value=4.5,
                                    marks={i: str(i) for i in range(1, 6)},
                                ),
                                html.Br(),
                                html.Label("Número de reseñas"),
                                dcc.Input(
                                    id="input-number_of_reviews",
                                    type="number",
                                    min=0,
                                    value=10,
                                    style={"width": "100%"},
                                ),
                                html.Br(), html.Br(),
                                html.Label("Reviews por mes"),
                                dcc.Input(
                                    id="input-reviews_per_month",
                                    type="number",
                                    min=0,
                                    step=0.1,
                                    value=0.5,
                                    style={"width": "100%"},
                                ),
                            ],
                        ),

                        html.H4("5. Anfitrión e ingresos"),
                        html.Div(
                            style={
                                "border": "1px solid #e3e6ee",
                                "borderRadius": "10px",
                                "padding": "10px 12px",
                            },
                            children=[
                                html.Label("Tasa de aceptación (%)"),
                                dcc.Input(
                                    id="input-host_acceptance_rate",
                                    type="number",
                                    min=0,
                                    max=100,
                                    value=95,
                                    style={"width": "100%"},
                                ),
                                html.Br(), html.Br(),
                                html.Label("Tasa de respuesta (%)"),
                                dcc.Input(
                                    id="input-host_response_rate",
                                    type="number",
                                    min=0,
                                    max=100,
                                    value=98,
                                    style={"width": "100%"},
                                ),
                                html.Br(), html.Br(),
                                html.Label("Ingresos estimados últimos 365 días"),
                                dcc.Input(
                                    id="input-estimated_revenue_l365d",
                                    type="number",
                                    min=0,
                                    value=5000,
                                    style={"width": "100%"},
                                ),
                            ],
                        ),

                        html.H4("6. Amenidades"),
                        html.Div(
                            style={
                                "border": "1px solid #e3e6ee",
                                "borderRadius": "10px",
                                "padding": "10px 12px",
                            },
                            children=[
                                dcc.Checklist(
                                    id="amenities",
                                    options=[
                                        {"label": "Wifi", "value": "Wifi"},
                                        {"label": "Cocina", "value": "Kitchen_and_dining"},
                                        {"label": "Aire acondicionado", "value": "Air_conditioning"},
                                        {"label": "Lavadora/Secadora", "value": "Washer_dryer"},
                                        {"label": "TV", "value": "TV"},
                                        {"label": "Caja fuerte", "value": "Safe"},
                                        {"label": "Refrigerador", "value": "Refrigerator"},
                                        {"label": "Detector de humo", "value": "Smoke_alarm_home_safety"},
                                        {"label": "Esenciales", "value": "Essentials"},
                                        {"label": "Otros servicios", "value": "Services"},
                                    ],
                                    value=[
                                        "Wifi",
                                        "Kitchen_and_dining",
                                        "TV",
                                        "Refrigerator",
                                        "Essentials",
                                    ],
                                    labelStyle={"display": "block", "fontSize": "13px"},
                                ),
                            ],
                        ),

                        html.Br(),

                        html.Button(
                            "Generar mapa de precios",
                            id="predict-button",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "backgroundColor": "#00CC96",
                                "border": "none",
                                "color": "white",
                                "padding": "10px",
                                "fontSize": "16px",
                                "fontWeight": "600",
                                "borderRadius": "10px",
                                "cursor": "pointer",
                            },
                        ),
                        html.Br(),
                    ],
                ),

               
                html.Div(
                    style={
                        "width": "65%",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "16px",
                    },
                    children=[
                        dcc.Graph(
                            id="price-map",
                            style={"height": "550px"},
                        ),
                        html.Div(
                            id="price-info",
                            style={
                                "backgroundColor": "#f9fafc",
                                "borderRadius": "10px",
                                "padding": "10px 14px",
                                "fontSize": "14px",
                            },
                        ),
                    ],
                ),
            ],
        )
    ],
)



def score_location_from_coords(lat, lon):
   
        
        center_lat, center_lon = 40.4168, -3.7038

        dist = np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)

       
        score = 5 - dist / 0.03
        return float(np.clip(score, 1.0, 5.0))

@app.callback(
    [Output("price-map", "figure"),
     Output("price-info", "children")],
    Input("predict-button", "n_clicks"),
    [
        State("property-type", "value"),
        State("input-bathrooms", "value"),
        State("input-beds", "value"),
        State("input-bedrooms", "value"),
        State("input-accommodates", "value"),
        State("input-rating", "value"),
        State("input-review_scores_cleanliness", "value"),
        State("input-review_scores_accuracy", "value"),
        State("input-review_scores_checkin", "value"),
        State("input-minimum_nights", "value"),
        State("input-maximum_nights", "value"),
        State("input-availability_365d", "value"),
        State("input-number_of_reviews", "value"),
        State("input-estimated_revenue_l365d", "value"),
        State("input-host_acceptance_rate", "value"),
        State("input-host_response_rate", "value"),
        State("input-reviews_per_month", "value"),
        State("amenities", "value"),
    ],
)
def generate_price_map(
    n_clicks,
    prop_type,
    bathrooms,
    beds,
    bedrooms,
    accommodates,
    rating,
    clean_score,
    acc_score,
    checkin_score,
    min_nights,
    max_nights,
    availability,
    num_reviews,
    est_revenue,
    host_accept_rate,
    host_response_rate,
    reviews_per_month,
    amenities,
):
    if n_clicks == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Presiona 'Generar Mapa' para ver predicciones",
            xaxis_title="Longitud",
            yaxis_title="Latitud",
            height=600,
        )
        return fig, "Esperando entrada..."

    # Rango de coordenadas
    lat_min, lat_max = 40.35, 40.50
    lon_min, lon_max = -3.75, -3.60

    lats = np.linspace(lat_min, lat_max, 30)
    lons = np.linspace(lon_min, lon_max, 30)

    # Amenidades -> dummies
    wifi = 1 if "Wifi" in amenities else 0
    kitchen = 1 if "Kitchen_and_dining" in amenities else 0
    air = 1 if "Air_conditioning" in amenities else 0
    washer_dryer = 1 if "Washer_dryer" in amenities else 0
    tv = 1 if "TV" in amenities else 0
    safe = 1 if "Safe" in amenities else 0
    fridge = 1 if "Refrigerator" in amenities else 0
    smoke_alarm = 1 if "Smoke_alarm_home_safety" in amenities else 0
    essentials = 1 if "Essentials" in amenities else 0
    services = 1 if "Services" in amenities else 0

    # Tipo de propiedad -> one hot
    property_entire = 1 if prop_type == "entire" else 0
    property_private = 1 if prop_type == "private" else 0
    property_shared = 1 if prop_type == "shared" else 0
    property_hotel = 1 if prop_type == "hotel" else 0
    property_other = 1 if prop_type == "other" else 0

    # Otras puntuaciones derivadas del rating general
    comm_score = rating
    value_score = rating

    predicciones = []
    coords = []

    for lat in lats:
        for lon in lons:
            # review_scores_location calculado a partir de lat/lon
            loc_score = score_location_from_coords(lat, lon)

            # estimated_occupancy_l365d aprox: 70% de días disponibles
            est_occupancy = availability * 0.7

            fila = [
                
                host_response_rate,        # host_response_rate
                host_accept_rate,          # host_acceptance_rate
                lat,                       # latitude
                lon,                       # longitude
                accommodates,              # accommodates
                bathrooms,                 # bathrooms
                bedrooms,                  # bedrooms
                beds,                      # beds
                min_nights,                # minimum_nights
                max_nights,                # maximum_nights
                availability,              # availability_365
                num_reviews,               # number_of_reviews
                est_occupancy,             # estimated_occupancy_l365d
                est_revenue,               # estimated_revenue_l365d
                rating,                    # review_scores_rating
                acc_score,                 # review_scores_accuracy
                clean_score,               # review_scores_cleanliness
                checkin_score,             # review_scores_checkin
                comm_score,                # review_scores_communication
                loc_score,                 # review_scores_location
                value_score,               # review_scores_value
                reviews_per_month,         # reviews_per_month
                wifi,                      # Wifi
                air,                       # Air_conditioning
                kitchen,                   # Kitchen_and_dining
                washer_dryer,              # Washer_dryer
                tv,                        # TV
                safe,                      # Safe
                fridge,                    # Refrigerator
                smoke_alarm,               # Smoke_alarm_home_safety
                essentials,                # Essentials
                services,                  # Services
                property_entire,           # property_Entire_Place
                property_hotel,            # property_Hotel_Room
                property_other,            # property_Other
                property_private,          # property_Private_Room
                property_shared,           # property_Shared_Room
            ]

            coords.append((lat, lon))
            x_row = np.array([fila], dtype="float32")
            precio_pred = float(
                modelo_regresion.predict(x_row, verbose=0)[0, 0]
            )
            predicciones.append(max(precio_pred, 7.0))

    df_map = pd.DataFrame({
        "latitude": [c[0] for c in coords],
        "longitude": [c[1] for c in coords],
        "precio_pred": predicciones,
    })

    fig = px.density_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        z="precio_pred",
        radius=20,
        center=dict(lat=(lat_min + lat_max) / 2, lon=(lon_min + lon_max) / 2),
        zoom=11,
        mapbox_style="carto-positron",
        color_continuous_scale="Viridis",
        title="Mapa de precios estimados (€/noche)",
    )

    precio_min = float(df_map["precio_pred"].min())
    precio_max = float(df_map["precio_pred"].max())
    precio_prom = float(df_map["precio_pred"].mean())

    info = html.Div([
        html.Div([
            html.Span("Precio Promedio: ", style={"fontWeight": "bold"}),
            html.Span(f"{precio_prom:.2f} €/noche", style={"color": "#4CAF50"}),
        ]),
        html.Div([
            html.Span("Precio Mínimo: ", style={"fontWeight": "bold"}),
            html.Span(f"{precio_min:.2f} €/noche"),
        ]),
        html.Div([
            html.Span("Precio Máximo: ", style={"fontWeight": "bold"}),
            html.Span(f"{precio_max:.2f} €/noche"),
        ]),
    ])

    return fig, info





if __name__ == "__main__":
    app.run(debug=True)


    

  




    

  



    

  
