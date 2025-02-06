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

st.markdown("<h1>🚀 Smart Route Optimization</h1>", unsafe_allow_html=True)
st.write("Optimize routes using Clustering & TSP with Google Maps API.")

# ---------------------- FOLDING BOX FOR FILE REQUIREMENTS ----------------------
with st.expander("📄 **Click to see File Requirements**"):
    st.markdown(
        """
        ### 📊 **Excel File Structure**
        The Excel file should have the following format:
        - The **sheet name** must be Sheet1.
        - The **headers** should start from the **2nd row** with these columns:
        
        | A | B | C | D | E | F | G | H | I |
        |---|---|---|---|---|---|---|---|---|
        |   |   |   |   |   |   |   |   |   |
        | 2 | Nombre Comercial | Calle | No. | Sector | Municipio | Provincia | **Latitud** | **Longitud** |
        | 3 | Example Name | Example St. | 123 | Sector 1 | City 1 | Province 1 | **18.1234** | **-69.9876** |
        | 4 | Example Name 2 | Another St. | 456 | Sector 2 | City 2 | Province 2 | **18.5678** | **-69.6543** |

        **⚠️ Important Note:**  
        - If Latitud and Longitud are **missing**, you can retrieve them using the **Google Maps API** in this platform.
        - The more accurate the coordinates, the better the clustering and routing results.

        """,
        unsafe_allow_html=True
    )

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

    df = apply_balanced_clustering(df, num_clusters, max_points_per_cluster)

    agent = st.number_input("🔍 Select Agent", 1, num_clusters, 1)
    cluster_data = df[df["Cluster"] == agent - 1].copy()

    # ---------------------- START & END POINT SELECTION ----------------------
    start_point = st.selectbox("🚀 Select Start Location", cluster_data["Nombre Comercial"].tolist())
    end_point = st.selectbox("🏁 Select End Location", cluster_data["Nombre Comercial"].tolist())

    if not cluster_data.empty:
        cluster_data["Original Index"] = cluster_data.index

        # Reorder based on selected start and end points
        start_idx = cluster_data[cluster_data["Nombre Comercial"] == start_point].index[0]
        end_idx = cluster_data[cluster_data["Nombre Comercial"] == end_point].index[0]

        tsp_order = [start_idx] + [i for i in cluster_data.index if i not in [start_idx, end_idx]] + [end_idx]
        cluster_data = cluster_data.loc[tsp_order].reset_index(drop=True)
        cluster_data["Order"] = range(1, len(cluster_data) + 1)

        # ---------------------- MAP DISPLAY WITH FIX ----------------------
        m = folium.Map(zoom_start=12)

        # Add markers and store bounds
        bounds = []
        for _, row in cluster_data.iterrows():
            loc = [row["Latitud"], row["Longitud"]]
            folium.Marker(loc, popup=f"{row['Nombre Comercial']}<br>Original Index: {row['Original Index']}").add_to(m)
            bounds.append(loc)

        folium.PolyLine(cluster_data[["Latitud", "Longitud"]].values, color="blue", weight=2.5, opacity=1).add_to(m)

        m.fit_bounds(bounds)  # Ensure full view of all points

        st.write(f"## 🌍 Route for Agent {agent}")
        st_folium(m, width=800, height=500)

        st.download_button("📥 Download Optimized Route", cluster_data.to_csv(index=False), f"agent_{agent}_optimized_route.csv", "text/csv")
