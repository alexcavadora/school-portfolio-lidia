import ssl
import certifi
import requests
import pandas as pd
import folium
import io

# Reemplaza 'your_api_key' con tu API Key real
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/097f850d6200d5352026c5e6fabe1d3c/MODIS_NRT/world/10/"

# Define los límites aproximados de California
CALIFORNIA_BOUNDS = {
    "min_lat": 32.0, "max_lat": 42.0,
    "min_lon": -125.0, "max_lon": -113.0
}
def get_fire_data():
    """Descarga los datos de incendios activos desde NASA FIRMS."""
    try:
        # response = requests.get(FIRMS_URL, verify="C:/Users/uif59266/Downloads/firms.modaps.eosdis.nasa.gov.pem", timeout=10)
        response = requests.get(FIRMS_URL, verify=False, timeout=10)
        response.raise_for_status()  # Lanza una excepción si hay error en la solicitud
        return response.text
    except requests.exceptions.RequestException as e:
        print("Error descargando los datos:", e)
        return None

def filter_california_fires(data):
    """Filtra los incendios dentro de California."""
    try:
        df = pd.read_csv(io.StringIO(data))  # Convierte el texto a un DataFrame
        df = df[
            (df["latitude"] >= CALIFORNIA_BOUNDS["min_lat"]) &
            (df["latitude"] <= CALIFORNIA_BOUNDS["max_lat"]) &
            (df["longitude"] >= CALIFORNIA_BOUNDS["min_lon"]) &
            (df["longitude"] <= CALIFORNIA_BOUNDS["max_lon"])
        ]
        return df
    except Exception as e:
        print("Error procesando los datos:", e)
        return pd.DataFrame()  # Devuelve un DataFrame vacío si hay error

def plot_fires_on_map(df):
    """Genera un mapa interactivo con los incendios."""
    if df.empty:
        print("No hay datos de incendios en California para visualizar.")
        return

    fire_map = folium.Map(location=[37.5, -119.5], zoom_start=6)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.7,
            popup=f"Brillo: {row.get('brightness', 'N/A')}, Fecha: {row.get('acq_date', 'N/A')}"
        ).add_to(fire_map)

    fire_map.save("california_fires_map.html")
    print("Mapa guardado como 'california_fires_map.html'. Ábrelo en tu navegador.")

# Ejecutar flujo de trabajo
data = get_fire_data()

if data:
    with open("firms_data.txt", "w", encoding="utf-8") as f:
        f.write(data)
    print("Datos descargados guardados en 'firms_data.txt'. Revisa el archivo para verificar el formato.")

    california_fires = filter_california_fires(data)
    plot_fires_on_map(california_fires)