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

# Apply black background & white text styling
st.markdown(
    """
    <style>
        body, .stApp { background-color: black !important; color: white !important; }
        h1, h2, h3, h4, h5, h6, p, label, .stButton>button, .stMarkdown { color: white !important; }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSlider>div>div>div>div, 
        .stSelectbox>div>div>div, .stDataFrame, .stTable, .stDownloadButton>button {
            background-color: #333333 !important;
            color: white !important;
            border-radius: 5px !important;
            border: 1px solid white !important;
        }
        .stDownloadButton>button {
            background-color: #444444 !important;
            color: white !important;
            border-radius: 5px !important;
            border: 1px solid white !important;
            font-size: 16px !important;
            font-weight: bold !important;
            padding: 10px !important;
        }
        .stDownloadButton>button:hover {
            background-color: #666666 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1> 🚀 Smart Route Optimization</h1>", unsafe_allow_html=True)
st.write("Optimize routes using Clustering & TSP with Google Maps API.")

# ---------------------- CORE FUNCTIONS ----------------------
def apply_balanced_clustering(df, num_clusters, max_points_per_cluster):
    """Applies balanced clustering to evenly distribute points near the set limit."""
    coords = df[["Latitud", "Longitud"]].values

    # Small noise added to avoid KMeans breaking due to identical points
    jitter = np.random.normal(0, 0.00001, coords.shape)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(coords + jitter)
    df["Cluster"] = kmeans.labels_

    # Group points into cluster lists
    clusters_dict = {i: df[df["Cluster"] == i].index.tolist() for i in range(num_clusters)}
    centroids = np.array(kmeans.cluster_centers_)

    max_iterations = 100  # Prevent infinite loops
    iterations = 0

    while iterations < max_iterations:
        overfilled_clusters = {c: len(p) for c, p in clusters_dict.items() if len(p) > max_points_per_cluster}
        underfilled_clusters = {c: len(p) for c, p in clusters_dict.items() if len(p) < max_points_per_cluster // 2}

        if not overfilled_clusters and not underfilled_clusters:
            break

        for cluster_id, point_count in overfilled_clusters.items():
            if point_count <= max_points_per_cluster:
                continue

            excess_points = clusters_dict[cluster_id][- (point_count - max_points_per_cluster):]
            clusters_dict[cluster_id] = clusters_dict[cluster_id][: max_points_per_cluster]

            for excess_point in excess_points:
                excess_coords = df.loc[excess_point, ["Latitud", "Longitud"]].values.reshape(1, -1).astype(float)
                distances = np.linalg.norm(centroids - excess_coords, axis=1)
                sorted_clusters = np.argsort(distances)
                nearest_cluster = next((c for c in sorted_clusters if c in underfilled_clusters), None)

                if nearest_cluster is not None:
                    clusters_dict[nearest_cluster].append(excess_point)

        iterations += 1

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

    df = df.astype({col: str for col in df.select_dtypes('object').columns})
    st.write("✅ **File uploaded successfully!** Preview:")
    st.dataframe(df.head())

    # ---------------------- PARAMETER SELECTION ----------------------
    col1, col2 = st.columns(2)
    with col1:
        num_clusters = st.slider("🔹 Number of Agents", 2, 20, 10)
    with col2:
        max_points_per_cluster = st.slider("🔹 Max Locations per Agent", 50, 500, 281)

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

    df = apply_balanced_clustering(df, num_clusters, max_points_per_cluster)

    agent = st.number_input("🔍 Select Agent", 1, num_clusters, 1)
    cluster_data = df[df["Cluster"] == agent - 1].copy()

    if not cluster_data.empty:
        cluster_data["Original Index"] = cluster_data.index
        tsp_order = tsp_nearest_neighbor(cluster_data[["Latitud", "Longitud"]].values)
        cluster_data = cluster_data.iloc[tsp_order].reset_index(drop=True)
        cluster_data["Order"] = range(1, len(cluster_data) + 1)

        m = folium.Map(location=cluster_data[["Latitud", "Longitud"]].mean().tolist(), zoom_start=12)
        for _, row in cluster_data.iterrows():
            folium.Marker(
                [row["Latitud"], row["Longitud"]],
                popup=f"{row['Nombre Comercial']}<br>Original Index: {row['Original Index']}"
            ).add_to(m)

        folium.PolyLine(
            cluster_data[["Latitud", "Longitud"]].values, color="blue", weight=2.5, opacity=1
        ).add_to(m)

        st.write(f"## 🗺️ Route for Agent {agent}")
        st_folium(m, width=800, height=500)

        # ---------------------- Styled CSV EXPORT BUTTON ----------------------
        csv_buffer = io.StringIO()
        cluster_data.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Download Optimized Route",
            csv_buffer.getvalue(),
            f"agent_{agent}_optimized_route.csv",
            "text/csv"
        )
