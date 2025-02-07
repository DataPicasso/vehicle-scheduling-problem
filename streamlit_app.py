import streamlit as st
import pandas as pd
import numpy as np
import folium
import streamlit.components.v1 as components  # Required for embedding HTML ads
from streamlit_folium import st_folium
import googlemaps
import io
from sklearn.cluster import KMeans
from geopy.distance import geodesic

# ---------------------- STREAMLIT APP SETUP ----------------------
st.set_page_config(page_title="🏎️ AI Route Optimization", layout="wide")


# ---------------------- BESTROUTES AI CUSTOM HEADER ----------------------
st.markdown(
    """
    <style>
        .best-routes-header {
            font-family: 'Arial', sans-serif;
            font-weight: bold;
            font-size: 36px;
            text-align: center;
            color: #FF0000;  /* Red Color */
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-top: 20px;
        }
        .ai-text {
            color: black;
            font-size: 28px;
        }
        .author-credit {
            font-family: 'Arial', sans-serif;
            font-size: 14px;
            text-align: center;
            color: #777; /* Gray color */
            margin-top: 5px;
        }
    </style>
    <div class="best-routes-header">
        BESTROUTES <span class="ai-text">AI</span>
    </div>
    <div class="author-credit">
        By: Pedro Miguel Figueroa Domínguez
    </div>
    """,
    unsafe_allow_html=True
)



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

st.markdown("<h1> 🏎️ AI Route Optimization</h1>", unsafe_allow_html=True)
st.write("Optimize routes using Clustering & TSP with Google Maps API.")

# ---------------------- FUNCTION TO GENERATE TEST DATA ----------------------
def get_test_data():
    test_data = pd.DataFrame({
        "Nombre Comercial": [
            "Colmado La Esquina", "Supermercado Bravo", "Ferretería Popular", 
            "Farmacia GBC", "Panadería El Buen Gusto"
        ],
        "Calle": [
            "Av. Winston Churchill", "Calle del Sol", "Av. España", 
            "Calle Duarte", "Av. Independencia"
        ],
        "No.": ["101", "202", "303", "404", "505"],
        "Sector": ["Piantini", "Centro Histórico", "Ensanche Ozama", "Zona Colonial", "Gazcue"],
        "Municipio": ["Santo Domingo", "Santiago", "Santo Domingo Este", "Distrito Nacional", "Distrito Nacional"],
        "Provincia": ["Distrito Nacional", "Santiago", "Santo Domingo", "Distrito Nacional", "Distrito Nacional"],
        "Latitud": [18.4663, 19.4517, 18.4821, 18.4703, 18.4559],
        "Longitud": [-69.9337, -70.6970, -69.8730, -69.8903, -69.9020]
    })
    return test_data
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

# ---------------------- USE TEST DATA BUTTON ----------------------
if st.button("📊 Usar CSV de Prueba"):
    df = get_test_data()
    st.success("✅ ¡Se cargó el dataset de prueba con ubicaciones reales de República Dominicana!")
else:
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    else:
        df = None

if df is not None:
    df['Latitud'] = df['Latitud'].astype(float)
    df['Longitud'] = df['Longitud'].astype(float)
    st.write("✅ **Dataset cargado correctamente**. Vista previa:")
    st.dataframe(df.head())
  # ---------------------- USE TEST DATA BUTTON ----------------------
if st.button("📊 Usar CSV de Prueba"):
    df = get_test_data()
    st.success("✅ ¡Se cargó el dataset de prueba con ubicaciones reales de República Dominicana!")
else:
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    else:
        df = None

if df is not None:
    # Ensure Latitud & Longitud are floats
    df["Latitud"] = df["Latitud"].astype(float)
    df["Longitud"] = df["Longitud"].astype(float)

    st.write("✅ **Dataset cargado correctamente**. Vista previa:")
    st.dataframe(df.head())

    # ---------------------- PARAMETER SELECTION ----------------------
    col1, col2 = st.columns(2)
    with col1:
        num_clusters = st.slider("🔹 Number of Agents", 2, min(len(df), 20), 5)
    with col2:
        max_points_per_cluster = st.slider("🔹 Max Locations per Agent", 2, len(df), max(5, len(df) // num_clusters))

    df = apply_balanced_clustering(df, num_clusters, max_points_per_cluster)

    agent = st.number_input("🔍 Select Agent ", 1, num_clusters, 1)
    cluster_data = df[df["Cluster"] == agent - 1].copy()

    # Ensure there are locations assigned to this agent
    if cluster_data.empty:
        st.warning("⚠️ No locations assigned to this agent. Try reducing the number of clusters.")
        st.stop()

    # ---------------------- START & END POINT SELECTION ----------------------
    if len(cluster_data) < 2:
        st.warning("⚠️ Not enough locations in this cluster to create a route.")
        st.stop()

    start_point = st.selectbox("🚶‍➡️ Select Start Location", cluster_data["Nombre Comercial"].tolist())
    end_point = st.selectbox("🚶 Select End Location", cluster_data["Nombre Comercial"].tolist())

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

    st.write(f"## 🧑🏽‍💼 Route for Agent {agent}")
    st_folium(m, width=800, height=500)

    st.download_button("📥 Download Optimized Route", cluster_data.to_csv(index=False), f"agent_{agent}_optimized_route.csv", "text/csv")
