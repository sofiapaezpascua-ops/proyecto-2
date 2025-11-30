import os
import warnings
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# caragar los modelos y datos necesarios

df = pd.read_csv("listings_limpio.csv")

#Definir la función de elu+1

def elu_plus_one(x):
    return tf.keras.activations.elu(x) + 1.0

# Modelo de regresión
modelo_regresion = tf.keras.models.load_model(
    "red neuronal_airbnb.h5",custom_objects={"elu_plus_one": elu_plus_one}, compile=False
)

# Modelo de clasificación
modelo_clasificacion = tf.keras.models.load_model(
    "modelo_clasificacion_andes.h5", compile=False
)


NEIGH_DEFAULT = df["neighbourhood_cleansed"].mode()[0]



app = dash.Dash(__name__)

# Layout del dashboard
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
                            "Primero se estima el precio por noche (regresión) y, más abajo, "
                            "la probabilidad de obtener una puntuación alta (clasificación).",
                            style={
                                "color": "#555",
                                "fontSize": "13px",
                                "marginBottom": "10px",
                            },
                        ),

                        # 1. Tipo de alojamiento
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

                        # 2. Capacidad y distribución
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

                        # 3. Estancia y ocupación
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

                        # 4. Reseñas y puntuaciones
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

                        # 5. Anfitrión e ingresos
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
                                html.Label("Precio actual por noche (€)"),
                                dcc.Input(
                                    id="input-price",
                                    type="number",
                                    min=0,
                                    value=100,
                                    style={"width": "100%"},
                                ),
                                html.Br(), html.Br(),
                                html.Label("Ingresos estimados últimos 365 días (€)"),
                                dcc.Input(
                                    id="input-estimated_revenue_l365d",
                                    type="number",
                                    min=0,
                                    value=5000,
                                    style={"width": "100%"},
                                ),
                            ],
                        ),

                        # 6. Amenidades
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

                        # 7. Datos necesarios para clasificación
                        html.H4(
                            "Predicción de rating (modelo de clasificación)",
                            style={"marginTop": "14px"},
                        ),
                        html.Div(
                            style={
                                "border": "1px solid #e3e6ee",
                                "borderRadius": "10px",
                                "padding": "10px 12px",
                            },
                            children=[
                                html.P(
                                    "Esta sección usa un segundo modelo para estimar la "
                                    "probabilidad de que el anuncio tenga rating ≥ 4.8.",
                                    style={"fontSize": "12px", "color": "#555"},
                                ),
                                html.Label("Latitud del anuncio"),
                                dcc.Input(
                                    id="input-latitude",
                                    type="number",
                                    value=40.42,
                                    step=0.0001,
                                    style={"width": "100%"},
                                ),
                                html.Br(), html.Br(),
                                html.Label("Longitud del anuncio"),
                                dcc.Input(
                                    id="input-longitude",
                                    type="number",
                                    value=-3.70,
                                    step=0.0001,
                                    style={"width": "100%"},
                                ),
                                html.P(
                                    "La probabilidad de cada clase se muestra en la gráfica "
                                    "de barras debajo del mapa.",
                                    style={
                                        "fontSize": "11px",
                                        "color": "#777",
                                        "marginTop": "8px",
                                    },
                                ),
                            ],
                        ),

                        html.Br(),

                        html.Button(
                            "Generar mapa y predicción",
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

                # Columna con mapa y grafico de barras
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
                        dcc.Graph(
                            id="class-probs-graph",
                            style={"height": "280px"},
                        ),
                    ],
                ),
            ],
        )
    ],
)


# CALLBACK: Red neuronal regresión



@app.callback(
    [Output("price-map", "figure"),
     Output("price-info", "children")],
    Input("predict-button", "n_clicks"),
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
            title="Presiona 'Generar mapa y predicción' para ver resultados",
            xaxis_title="Longitud",
            yaxis_title="Latitud",
            height=600,
        )
        return fig, "Esperando entrada..."

    if amenities is None:
        amenities = []

    # Amenidades -> dummies
    def flag(name):
        return 1 if name in amenities else 0

    wifi = flag("Wifi")
    kitchen = flag("Kitchen_and_dining")
    air = flag("Air_conditioning")
    washer_dryer = flag("Washer_dryer")
    tv = flag("TV")
    safe = flag("Safe")
    fridge = flag("Refrigerator")
    smoke_alarm = flag("Smoke_alarm_home_safety")
    essentials = flag("Essentials")
    services = flag("Services")

    # Tipo de propiedad -> one hot
    property_entire = 1 if prop_type == "entire" else 0
    property_private = 1 if prop_type == "private" else 0
    property_shared = 1 if prop_type == "shared" else 0
    property_hotel = 1 if prop_type == "hotel" else 0
    property_other = 1 if prop_type == "other" else 0

    # Algunas puntuaciones derivadas
    comm_score = rating
    value_score = rating

    # Rango de coordenadas para el mapa (puedes ajustarlo a tu ciudad)
    lat_min, lat_max = 40.35, 40.50
    lon_min, lon_max = -3.75, -3.60

    lats = np.linspace(lat_min, lat_max, 30)
    lons = np.linspace(lon_min, lon_max, 30)

    predicciones = []
    coords = []

    for lat in lats:
        for lon in lons:
            # Estimación simple de ocupación (si quieres puedes mejorarla)
            est_occupancy = availability * 0.7

            fila = [
                # 37 features (orden consistente con el modelo de regresión)
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
                rating,                    # review_scores_location (aprox)
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
            html.Span("Precio promedio: ", style={"fontWeight": "bold"}),
            html.Span(f"{precio_prom:.2f} €/noche", style={"color": "#4CAF50"}),
        ]),
        html.Div([
            html.Span("Precio mínimo: ", style={"fontWeight": "bold"}),
            html.Span(f"{precio_min:.2f} €/noche"),
        ]),
        html.Div([
            html.Span("Precio máximo: ", style={"fontWeight": "bold"}),
            html.Span(f"{precio_max:.2f} €/noche"),
        ]),
    ])

    return fig, info


#Callback modelo de clasificación con redes neuronales

@app.callback(
    Output("class-probs-graph", "figure"),
    Input("predict-button", "n_clicks"),
    State("input-host_response_rate", "value"),
    State("input-host_acceptance_rate", "value"),
    State("input-accommodates", "value"),
    State("input-bathrooms", "value"),
    State("input-bedrooms", "value"),
    State("input-beds", "value"),
    State("input-minimum_nights", "value"),
    State("input-maximum_nights", "value"),
    State("input-availability_365d", "value"),
    State("input-number_of_reviews", "value"),
    State("input-rating", "value"),
    State("input-review_scores_accuracy", "value"),
    State("input-review_scores_cleanliness", "value"),
    State("input-review_scores_checkin", "value"),
    State("input-reviews_per_month", "value"),
    State("amenities", "value"),
    State("input-price", "value"),
    State("input-estimated_revenue_l365d", "value"),
    State("input-latitude", "value"),
    State("input-longitude", "value"),
    prevent_initial_call=True,
)
def actualizar_probabilidades(
    n_clicks,
    host_response_rate,
    host_acceptance_rate,
    accommodates,
    bathrooms,
    bedrooms,
    beds,
    minimum_nights,
    maximum_nights,
    availability_365,
    number_of_reviews,
    review_scores_rating,
    review_scores_accuracy,
    review_scores_cleanliness,
    review_scores_checkin,
    reviews_per_month,
    amenities,
    price_noche,
    estimated_revenue_l365d,
    latitude,
    longitude,
):
    if amenities is None:
        amenities = []

    def flag(name):
        return 1 if name in amenities else 0

    wifi = flag("Wifi")
    air_conditioning = flag("Air_conditioning")
    kitchen_and_dining = flag("Kitchen_and_dining")
    washer_dryer = flag("Washer_dryer")
    tv = flag("TV")
    safe = flag("Safe")
    refrigerator = flag("Refrigerator")
    smoke_alarm = flag("Smoke_alarm_home_safety")
    essentials = flag("Essentials")
    services = flag("Services")

    # Derivados simples para las columnas que espera el modelo
    estimated_occupancy_l365d = availability_365 * 0.7
    review_scores_communication = review_scores_rating
    review_scores_location = review_scores_rating
    review_scores_value = review_scores_rating

    # Diccionario EXACTO con las columnas del modelo de clasificación
    sample = {
        "id": np.array([0], dtype="int64"),
        "host_response_rate": np.array([host_response_rate], dtype="float32"),
        "host_acceptance_rate": np.array([host_acceptance_rate], dtype="float32"),
        "latitude": np.array([latitude], dtype="float32"),
        "longitude": np.array([longitude], dtype="float32"),
        "accommodates": np.array([accommodates], dtype="float32"),
        "bathrooms": np.array([bathrooms], dtype="float32"),
        "bedrooms": np.array([bedrooms], dtype="float32"),
        "beds": np.array([beds], dtype="float32"),
        "price": np.array([price_noche], dtype="float32"),
        "minimum_nights": np.array([minimum_nights], dtype="float32"),
        "maximum_nights": np.array([maximum_nights], dtype="float32"),
        "availability_365": np.array([availability_365], dtype="float32"),
        "number_of_reviews": np.array([number_of_reviews], dtype="float32"),
        "estimated_occupancy_l365d": np.array([estimated_occupancy_l365d], dtype="float32"),
        "estimated_revenue_l365d": np.array([estimated_revenue_l365d], dtype="float32"),
        "review_scores_rating": np.array([review_scores_rating], dtype="float32"),
        "review_scores_accuracy": np.array([review_scores_accuracy], dtype="float32"),
        "review_scores_cleanliness": np.array([review_scores_cleanliness], dtype="float32"),
        "review_scores_checkin": np.array([review_scores_checkin], dtype="float32"),
        "review_scores_communication": np.array([review_scores_communication], dtype="float32"),
        "review_scores_location": np.array([review_scores_location], dtype="float32"),
        "review_scores_value": np.array([review_scores_value], dtype="float32"),
        "reviews_per_month": np.array([reviews_per_month], dtype="float32"),
        "Wifi": np.array([wifi], dtype="int64"),
        "Air_conditioning": np.array([air_conditioning], dtype="int64"),
        "Kitchen_and_dining": np.array([kitchen_and_dining], dtype="int64"),
        "Washer_dryer": np.array([washer_dryer], dtype="int64"),
        "TV": np.array([tv], dtype="int64"),
        "Safe": np.array([safe], dtype="int64"),
        "Refrigerator": np.array([refrigerator], dtype="int64"),
        "Smoke_alarm_home_safety": np.array([smoke_alarm], dtype="int64"),
        "Essentials": np.array([essentials], dtype="int64"),
        "Services": np.array([services], dtype="int64"),
        "neighbourhood_cleansed": np.array([NEIGH_DEFAULT], dtype="str"),
    }

    p1 = float(modelo_clasificacion.predict(sample, verbose=0)[0, 0])
    p0 = 1.0 - p1

    fig = px.bar(
        x=["Rating < 4.8", "Rating ≥ 4.8"],
        y=[p0, p1],
        labels={"x": "Clase", "y": "Probabilidad"},
        range_y=[0, 1],
    )
    fig.update_layout(
        title="Probabilidad de alta puntuación (clasificación)",
        yaxis_tickformat=".0%",
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True)


    

  



    

  
