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
def apply_balanced_clustering(df, num_clusters, max_points_per_cluster):
    """Applies balanced clustering by redistributing excess points to maintain equal clusters."""
    coords = df[["Latitud", "Longitud"]].values

    # Small noise added to avoid KMeans breaking due to identical points
    jitter = np.random.normal(0, 0.00001, coords.shape)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(coords + jitter)
    df["Cluster"] = kmeans.labels_

    # Group points into cluster lists
    clusters_dict = {i: df[df["Cluster"] == i].index.tolist() for i in range(num_clusters)}
    centroids = np.array(kmeans.cluster_centers_)

    # Balance clusters by redistributing excess points
    max_iterations = 100  # Prevent infinite loops
    iterations = 0

    for cluster_id, points in clusters_dict.items():
        while len(points) > max_points_per_cluster and iterations < max_iterations:
            excess_point = points.pop()

            # Ensure excess_coords is a 2D NumPy array
            excess_coords = df.loc[excess_point, ["Latitud", "Longitud"]].values.reshape(1, -1).astype(float)

            # Compute distances
            distances = np.linalg.norm(centroids - excess_coords, axis=1)

            # Find nearest cluster (excluding the same one)
            nearest_cluster = np.argsort(distances)[1]

            # Move the excess point
            clusters_dict[nearest_cluster].append(excess_point)
            iterations += 1

    # Update cluster assignments
    new_labels = np.zeros(len(df), dtype=int)
    for cluster_id, indices in clusters_dict.items():
        new_labels[indices] = cluster_id
    df["Cluster"] = new_labels

    return df

def tsp_nearest_neighbor(points):
    """Solves TSP using nearest neighbor heuristic."""
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
        max_points_per_cluster = st.slider("🔹 Max Locations per Agent", 10, 500, 50)

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
    df = apply_balanced_clustering(df, num_clusters, max_points_per_cluster)

    # ---------------------- ROUTE VISUALIZATION ----------------------
    agent = st.number_input("🔍 Select Agent", 1, num_clusters, 1)
    cluster_data = df[df["Cluster"] == agent - 1].copy()

    if not cluster_data.empty:
        # Save original order before applying TSP
        cluster_data["Original Index"] = cluster_data.index

        # Get optimized order using TSP
        tsp_order = tsp_nearest_neighbor(cluster_data[["Latitud", "Longitud"]].values)
        
        # Apply TSP order
        cluster_data = cluster_data.iloc[tsp_order].reset_index(drop=True)
        cluster_data["Order"] = range(1, len(cluster_data) + 1)

        # ---------------------- DISPLAY ALL POINTS BEFORE ROUTE ----------------------
        m = folium.Map(location=cluster_data[["Latitud", "Longitud"]].mean().tolist(), zoom_start=12)

        # Add all locations to the map
        for _, row in cluster_data.iterrows():
            folium.Marker(
                [row["Latitud"], row["Longitud"]],
                popup=f"{row['Nombre Comercial']}<br>Original Index: {row['Original Index']}"
            ).add_to(m)

        # Draw route
        folium.PolyLine(
            cluster_data[["Latitud", "Longitud"]].values, color="blue", weight=2.5, opacity=1
        ).add_to(m)

        # Display map
        st.write(f"## 🗺️ Route for Agent {agent}")
        st_folium(m, width=800, height=500)

        # ---------------------- CSV EXPORT ----------------------
        csv_buffer = io.StringIO()
        cluster_data.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Download Optimized Route",
            csv_buffer.getvalue(),
            f"agent_{agent}_optimized_route.csv",
            "text/csv"
        )
