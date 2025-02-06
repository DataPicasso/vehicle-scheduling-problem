import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import googlemaps
import io
from sklearn.cluster import KMeans
from geopy.distance import geodesic

# ---------------------- STREAMLIT APP SETUP ----------------------
st.set_page_config(page_title="🚀 Smart Route Optimization", layout="wide")
st.markdown("<h1>📍 Smart Route Optimization</h1>", unsafe_allow_html=True)
st.write("Optimize routes using Clustering & TSP with Google Maps API.")

# ---------------------- CORE FUNCTIONS ----------------------
def apply_clustering(df, num_clusters, max_points_per_cluster):
    """Cluster locations using KMeans with distance matrix optimization."""
    coords = df[["Latitud", "Longitud"]].values
    
    # Add small noise to prevent identical points from breaking KMeans
    jitter = np.random.normal(0, 0.00001, coords.shape)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(coords + jitter)
    
    df["Cluster"] = kmeans.labels_
    return df

def tsp_nearest_neighbor(points):
    """Solve TSP using nearest neighbor heuristic with geodesic distances."""
    if len(points) < 2:
        return [0] if len(points) == 1 else []
    
    remaining = list(range(1, len(points)))
    route = [0]
    
    while remaining:
        last = route[-1]
        nearest = min(remaining, key=lambda x: geodesic(points[last], points[x]).meters)
        route.append(nearest)
        remaining.remove(nearest)
    
    return route

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    
    # Fix Arrow serialization by ensuring string types
    df = df.astype({col: str for col in df.select_dtypes('object').columns})
    
    st.write("✅ **File uploaded successfully!** Preview:")
    st.dataframe(df.head())

    # ---------------------- PARAMETER SELECTION ----------------------
    col1, col2 = st.columns(2)
    with col1:
        num_clusters = st.slider("🔹 Number of Agents", 2, 20, 10)
    with col2:
        max_points = st.slider("🔹 Max Locations per Agent", 50, 500, 281)

    # ---------------------- COORDINATE VALIDATION ----------------------
    if {"Latitud", "Longitud"}.issubset(df.columns):
        df = df.dropna(subset=["Latitud", "Longitud"])
    else:
        st.warning("Coordinates missing! Enter Google Maps API Key to geocode addresses.")
        api_key = st.text_input("🔑 Google Maps API Key")
        
        if api_key and "Ubicacion" in df.columns:
            gmaps = googlemaps.Client(key=api_key)
            df[["Latitud", "Longitud"]] = df["Ubicacion"].apply(
                lambda x: pd.Series(gmaps.geocode(x)[0]["geometry"]["location"]) if x else (np.nan, np.nan)
            )
            df = df.dropna(subset=["Latitud", "Longitud"])

    # ---------------------- CLUSTERING ----------------------
    df = apply_clustering(df, num_clusters, max_points)
    
    # ---------------------- ROUTE VISUALIZATION ----------------------
    agent = st.number_input("🔍 Select Agent", 1, num_clusters, 1)
    cluster_data = df[df["Cluster"] == agent-1].copy()
    
    if not cluster_data.empty:
        route_order = tsp_nearest_neighbor(cluster_data[["Latitud", "Longitud"]].values)
        cluster_data = cluster_data.iloc[route_order]
        cluster_data["Order"] = range(1, len(cluster_data)+1)
        
        # Map Visualization
        m = folium.Map(location=cluster_data[["Latitud", "Longitud"]].mean().tolist(), zoom_start=12)
        for _, row in cluster_data.iterrows():
            folium.Marker(
                [row["Latitud"], row["Longitud"]],
                popup=f"{row['Nombre Comercial']}<br>Order: {row['Order']}"
            ).add_to(m)
        
        # Display
        st.write(f"## 🗺️ Route for Agent {agent}")
        st_folium(m, width=800, height=500)
        st.download_button(
            "📥 Download Route",
            cluster_data.to_csv(index=False),
            f"agent_{agent}_route.csv",
            "text/csv"
        )
