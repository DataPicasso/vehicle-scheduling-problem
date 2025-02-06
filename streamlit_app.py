import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import googlemaps
import io
import nbformat
import requests
import os
import subprocess
from sklearn.cluster import KMeans

# ---------------------- STREAMLIT APP SETUP ----------------------
st.set_page_config(page_title="🚀 Smart Route Optimization", layout="wide")

st.markdown("<h1>📍 Smart Route Optimization</h1>", unsafe_allow_html=True)
st.write("Optimize routes using Clustering & TSP with Google Maps API.")

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    st.write("✅ **File uploaded successfully!** Preview:")
    st.dataframe(df.head())

    # ---------------------- PARAMETER SELECTION ----------------------
    col1, col2 = st.columns(2)
    with col1:
        num_clusters = st.slider("🔹 Number of Agents", min_value=2, max_value=20, value=10)
    with col2:
        max_points_per_cluster = st.slider("🔹 Places per Agent (Max Points per Cluster)", min_value=50, max_value=500, value=281)

    # ---------------------- HANDLE MISSING COORDINATES ----------------------
    if "Latitud" not in df.columns or "Longitud" not in df.columns:
        st.warning("Your dataset does not contain latitude and longitude. Enter your **Google Maps API Key** to extract coordinates.")

        api_key = st.text_input("🔑 Enter Google Maps API Key")
        if api_key:
            gmaps = googlemaps.Client(key=api_key)

            @st.cache_data
            def geocode_address(address):
                try:
                    result = gmaps.geocode(address)
                    location = result[0]["geometry"]["location"]
                    return location["lat"], location["lng"]
                except:
                    return None, None
            
            if "Ubicacion" in df.columns:
                df["Latitud"], df["Longitud"] = zip(*df["Ubicacion"].apply(geocode_address))
                df.dropna(subset=["Latitud", "Longitud"], inplace=True)
                st.success("✅ Coordinates extracted successfully!")
            else:
                st.error("❌ Missing 'Ubicacion' column. Cannot extract coordinates.")

    # ---------------------- DOWNLOAD & EXECUTE NOTEBOOK ----------------------
    st.write("🔄 **Fetching and Running VSP Notebook from GitHub...**")

    notebook_url = "https://raw.githubusercontent.com/DataPicasso/vehicle-scheduling-problem/main/VSP.ipynb"
    notebook_path = "/tmp/VSP.ipynb"
    script_path = "/tmp/VSP_script.py"

    apply_clustering = None  # Placeholder for the function
    tsp_nearest_neighbor = None  # Placeholder for TSP function

    try:
        # Download latest notebook
        response = requests.get(notebook_url)
        response.raise_for_status()

        # Save notebook locally
 
