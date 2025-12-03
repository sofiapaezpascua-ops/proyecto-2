# syntax=docker/dockerfile:1 
FROM ubuntu:22.04

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install flask==2.2.*
RUN pip install dash pandas numpy scikit-learn tensorflow

# 3. Carpeta de trabajo dentro del contenedor
WORKDIR /app

# 4. Copiar archivos de la app
COPY dashboard_airbnb.py .
COPY modelo_airbnb.keras .
COPY red_clasificacion.keras .
COPY listings_limpio.csv .

# 5. Puerto interno donde corre Dash
EXPOSE 8050

# 6. Comando para arrancar el dashboard
CMD ["python3", "dashboard_airbnb.py"]