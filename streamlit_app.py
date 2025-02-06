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
st.set_page_config(page_title="🚀 AI Route Optimization", layout="wide")

# Apply light Apple-like design
st.markdown(
    """
    <style>
        body, .stApp { background-color: #f7f7f7 !important; color: #333 !important; }
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stButton>button { color: #222 !important; }
        
        /* Inputs, dropdowns, and buttons */
        .stTextInput>div>div>input, .stNumberInput>div>div>input, 
        .stSlider>div>div>div>div, .stSelectbox>div>div>div,
        .stDataFrame, .stTable, .stDownloadButton>button {
            background-color: #ffffff !important;
            color: #333 !important;
            border-radius: 10px !important;
            border: 1px solid #ddd !important;
            box-shadow: 0px 2px 5px rgba(0,0,0,0.1) !important;
            padding: 8px !important;
        }

        /* Special styling for buttons */
        .stDownloadButton>button {
            background-color: #0084ff !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            font-size: 16px !important;
            font-weight: bold !important;
            padding: 10px 16px !important;
        }
        .stDownloadButton>button:hover {
            background-color: #0066cc !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>🚀 AI Route Optimization</h1>", unsafe_allow_html=True)
st.write("Optimize routes using Clustering & TSP with Google Maps API.")

# ---------------------- CORE FUNCTIONS ----------------------
def apply_balanced_clustering(df, num_clusters, max_points_per_cluster):
    """Apply clustering to evenly distribute points among agents."""
    coords = df[["Latitud", "Longitud"]].values
    jitter = np.random.normal(0, 0.00001, coords.shape)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(coords + jitter)
    df["Cluster"] = kmeans.labels_
    
    # Balance clusters
    clusters_dict = {i: df[df["Cluster"] == i].index.tolist() for i in range(num_clusters)}
    centroids = np.array(kmeans.cluster_centers_)
    max_iterations = 100
    iterations = 0
    
    while iterations < max_iterations:
        overfilled = {c: len(p) for c, p in clusters_dict.items() if len(p) > max_points_per_cluster}
        underfilled = {c: len(p) for c, p in clusters_dict.items() if len(p) < max_points_per_cluster // 2}
        
        if not overfilled and not underfilled:
            break
        
        for cluster_id, point_count in overfilled.items():
            if point_count <= max_points_per_cluster:
                continue
            excess_points = clusters_dict[cluster_id][- (point_count - max_points_per_cluster):]
            clusters_dict[cluster_id] = clusters_dict[cluster_id][: max_points_per_cluster]
            
            for excess_point in excess_points:
                excess_coords = df.loc[excess_point, ["Latitud", "Longitud"]].values.reshape(1, -1).astype(float)
                distances = np.linalg.norm(centroids - excess_coords, axis=1)
                sorted_clusters = np.argsort(distances)
                nearest_cluster = next((c for c in sorted_clusters if c in underfilled), None)
                
                if nearest_cluster is not None:
                    clusters_dict[nearest_cluster].append(excess_point)
        
        iterations += 1
    
    new_labels = np.zeros(len(df), dtype=int)
    for cluster_id, indices in clusters_dict.items():
        new_labels[indices] = cluster_id
    df["Cluster"] = new_labels
    
    return df

def tsp_nearest_neighbor(points, start_idx=0, end_idx=None):
    """Solves TSP using nearest neighbor heuristic with custom start and end."""
    if len(points) < 2:
        return [0] if len(points) == 1 else []
    
    remaining = list(range(len(points)))
    route = [start_idx]
    remaining.remove(start_idx)

    while remaining:
        last = route[-1]
        nearest = min(remaining, key=lambda x: geodesic(points[last], points[x]).meters)
        route.append(nearest)
        remaining.remove(nearest)

    if end_idx is not None and end_idx in route:
        route.remove(end_idx)
        route.append(end_idx)

    return route

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df = df.astype({col: str for col in df.select_dtypes('object').columns})
    st.write("✅ **File uploaded successfully!** Preview:")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        num_clusters = st.slider("🔹 Number of Agents", 2, 20, 10)
    with col2:
        max_points_per_cluster = st.slider("🔹 Max Locations per Agent", 50, 500, 281)
    
    df = apply_balanced_clustering(df, num_clusters, max_points_per_cluster)
    
    agent = st.number_input("🔍 Select Agent", 1, num_clusters, 1)
    cluster_data = df[df["Cluster"] == agent - 1].copy()

    if not cluster_data.empty:
        start_location = st.selectbox("🏁 Select Start Location", cluster_data["Nombre Comercial"].unique())
        end_location = st.selectbox("🏁 Select End Location", cluster_data["Nombre Comercial"].unique())

        start_idx = cluster_data[cluster_data["Nombre Comercial"] == start_location].index[0]
        end_idx = cluster_data[cluster_data["Nombre Comercial"] == end_location].index[0]

        tsp_order = tsp_nearest_neighbor(cluster_data[["Latitud", "Longitud"]].values, start_idx, end_idx)
        cluster_data = cluster_data.iloc[tsp_order].reset_index(drop=True)
        cluster_data["Order"] = range(1, len(cluster_data) + 1)

        # ---------------------- MAP DISPLAY ----------------------
        m = folium.Map(location=cluster_data[["Latitud", "Longitud"]].mean().tolist(), zoom_start=12)
        for _, row in cluster_data.iterrows():
            folium.Marker(
                [row["Latitud"], row["Longitud"]],
                popup=f"{row['Nombre Comercial']}<br>Order: {row['Order']}"
            ).add_to(m)

        folium.PolyLine(
            cluster_data[["Latitud", "Longitud"]].values, color="blue", weight=2.5, opacity=1
        ).add_to(m)

        st.write(f"## 🌍 Route for Agent {agent}")
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
