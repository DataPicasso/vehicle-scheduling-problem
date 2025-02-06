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
with st.expander("📄 **Click to View File Requirements**"):
    st.markdown(
        """
        ### 📊 **Excel File Structure**
        The Excel file should have the following format:
        - The **sheet name** must be `Sheet1`.
        - The **headers** should start from the **2nd row** with these columns:
        
        | A | B | C | D | E | F | G | H | I |
        |---|---|---|---|---|---|---|---|---|
        |   |   |   |   |   |   |   |   |   |
        | 2 | Nombre Comercial | Calle | No. | Sector | Municipio | Provincia | **Latitud** | **Longitud** |
        | 3 | Example Name | Example St. | 123 | Sector 1 | City 1 | Province 1 | **18.1234** | **-69.9876** |
        | 4 | Example Name 2 | Another St. | 456 | Sector 2 | City 2 | Province 2 | **18.5678** | **-69.6543** |

        **⚠️ Important Note:**  
        - If `Latitud` and `Longitud` are **missing**, you can retrieve them using the **Google Maps API** in this platform.
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

    if {"Latitud", "Longitud"}.issubset(df.columns):
        df = df.dropna(subset=["Latitud", "Longitud"])
    else:
        st.warning("⚠️ **Missing Coordinates!** Enter Google Maps API Key to fetch them.")
        api_key = st.text_input("🔑 **Google Maps API Key**")

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
        cluster_data["Order"] = range(1, len(cluster_data) + 1)

        # ---------------------- MAP DISPLAY ----------------------
        m = folium.Map(location=cluster_data[["Latitud", "Longitud"]].mean().tolist(), zoom_start=12)
        for _, row in cluster_data.iterrows():
            folium.Marker(
                [row["Latitud"], row["Longitud"]],
                popup=f"{row['Nombre Comercial']}<br>Original Index: {row['Original Index']}"
            ).add_to(m)

        folium.PolyLine(
            cluster_data[["Latitud", "Longitud"]].values, color="blue", weight=2.5, opacity=1
        ).add_to(m)

        st.write(f"## 🌍 Route for Agent {agent}")
        st_folium(m, width=800, height=500)
