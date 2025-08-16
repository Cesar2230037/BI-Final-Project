import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2

# Desactivar advertencias para una salida más limpia
import warnings
warnings.filterwarnings('ignore')

# Cargar el archivo CSV
try:
    df = pd.read_csv('dataset_procesado.csv')
    print("Datos cargados exitosamente.")
except FileNotFoundError:
    print("Error: El archivo 'dataset_procesado.csv' no se encontró.")
    exit()

def haversine(lon1, lat1, lon2, lat2):
    """
    Calcula la distancia de Haversine en kilómetros entre dos puntos.
    """
    R = 6371  # Radio de la Tierra en kilómetros
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def run_clustering_and_visualize(data, title, n_clusters=3):
    """
    Escala los datos, aplica K-means, calcula el Silhouette Score y visualiza los resultados.
    """
    print(f"\n--- Agrupamiento por {title} ---")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    score = silhouette_score(scaled_data, clusters)
    print(f"Puntuación de la Silueta para {title}: {score:.4f}")
    
    return clusters, scaled_data

# --- Agrupamiento por Distancia ---
df['distancia_km'] = df.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], 
                                                      row['dropoff_longitude'], row['dropoff_latitude']), axis=1)
clusters_distancia, scaled_distancia = run_clustering_and_visualize(df[['distancia_km']].values, 'Distancia')
df['cluster_distancia'] = clusters_distancia

# Visualización para Distancia
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
sns.histplot(x=df['distancia_km'], hue=df['cluster_distancia'], palette='viridis', multiple='stack', kde=True)
plt.title('Histograma de Distancia con Clústeres', fontsize=16)
plt.xlabel('Distancia (km)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
#plt.show()
plt.savefig('Histograma_de_Distancia_con_Clusteres.jpg')

# --- Agrupamiento por Duración ---
clusters_duracion, scaled_duracion = run_clustering_and_visualize(df[['trip_duration']].values, 'Duración')
df['cluster_duracion'] = clusters_duracion

# Visualización para Duración
plt.figure(figsize=(10, 6))
sns.histplot(x=df['trip_duration'], hue=df['cluster_duracion'], palette='viridis', multiple='stack', kde=True)
plt.title('Histograma de Duración con Clústeres', fontsize=16)
plt.xlabel('Duración del Viaje (segundos)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
#plt.show()
plt.savefig('Histograma_de_Duracion_con_Clusteres.jpg')

# --- Agrupamiento por Zona ---
zone_data = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].values
clusters_zona, scaled_zona = run_clustering_and_visualize(zone_data, 'Zona', n_clusters=4)
df['cluster_zona'] = clusters_zona

# Visualización para Zona
plt.figure(figsize=(12, 8))
sns.scatterplot(x='pickup_longitude', y='pickup_latitude', hue=clusters_zona, palette='viridis', s=50, alpha=0.6, data=df)
plt.title('Mapa de Dispersión de Puntos de Recogida por Clúster', fontsize=16)
plt.xlabel('Longitud', fontsize=12)
plt.ylabel('Latitud', fontsize=12)
#plt.show()
plt.savefig('Mapa_de_Dispersion_de_Puntos_de_Recogida_por_Cluster.jpg')

plt.figure(figsize=(12, 8))
sns.scatterplot(x='dropoff_longitude', y='dropoff_latitude', hue=clusters_zona, palette='viridis', s=50, alpha=0.6, data=df)
plt.title('Mapa de Dispersión de Puntos de Bajada por Clúster', fontsize=16)
plt.xlabel('Longitud', fontsize=12)
plt.ylabel('Latitud', fontsize=12)
#plt.show()
plt.savefig('Mapa_de_Dispercion_de_Puntos_de_Bajada_por_Cluster.jpg')

# --- Recomendaciones de Negocio ---
print("--- Análisis de Hallazgos y Recomendaciones para la Compañía de Taxis ---")

# Asignación de nombres de clústeres por zona
zone_cluster_col = 'cluster_zona'
zone_mapping = {
    df[zone_cluster_col].value_counts().idxmax(): 'Zona Central',
    df[zone_cluster_col].value_counts().index[1]: 'Zona de Tránsito',
    df[zone_cluster_col].value_counts().index[2]: 'Zona Aeropuerto/Periferia',
    df[zone_cluster_col].value_counts().index[3]: 'Zona Residencial'
}
df['nombre_zona'] = df[zone_cluster_col].map(zone_mapping)

# Asignación de nombres de clústeres por duración
duration_cluster_col = 'cluster_duracion'
duration_means = df.groupby(duration_cluster_col)['trip_duration'].mean().sort_values()
duration_mapping = {
    duration_means.index[0]: 'Viaje Corto',
    duration_means.index[1]: 'Viaje Medio',
    duration_means.index[2]: 'Viaje Largo'
}
df['tipo_viaje'] = df[duration_cluster_col].map(duration_mapping)

# Resumen de clústeres
print("\nCaracterísticas promedio de los clústeres de Distancia:")
distancia_summary = df.groupby('cluster_distancia')['distancia_km'].mean()
print(distancia_summary)

print("\nCaracterísticas promedio de los clústeres de Duración:")
duracion_summary = df.groupby('cluster_duracion')['trip_duration'].mean()
print(duracion_summary)

print("\nCentroides de los clústeres de Zona:")
zona_summary_pickup = df.groupby('cluster_zona')[['pickup_longitude', 'pickup_latitude']].mean()
print("\nCentroides de Puntos de Recogida:")
print(zona_summary_pickup)

print("\nAnálisis de la distribución de viajes por zona y duración:")
zone_duration_counts = df.groupby(['nombre_zona', 'tipo_viaje']).size().unstack(fill_value=0)
print(zone_duration_counts)
