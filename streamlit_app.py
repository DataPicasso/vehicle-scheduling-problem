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
from sklearn.metrics import pairwise_distances
from geopy.distance import geodesic

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

    apply_clustering = None
    tsp_nearest_neighbor = None

    try:
        response = requests.get(notebook_url)
        response.raise_for_status()

        with open(notebook_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        subprocess.run(
            ["jupyter", "nbconvert", "--to", "script", "--output", script_path.replace(".py", ""), notebook_path],
            check=True
        )

        converted_script_path = script_path
        if not os.path.exists(converted_script_path):
            st.error(f"🚨 Converted script not found: {converted_script_path}")
            raise FileNotFoundError(f"🚨 Converted script not found: {converted_script_path}")

        with open(converted_script_path, "r", encoding="utf-8") as script_file:
            script_content = script_file.read()

        exec_globals = {}
        exec(script_content, exec_globals)

        apply_clustering = exec_globals.get("apply_clustering")
        tsp_nearest_neighbor = exec_globals.get("tsp_nearest_neighbor")

        if not apply_clustering or not tsp_nearest_neighbor:
            raise ValueError("❌ Required functions not found in the notebook.")

        st.success("✅ **Notebook executed successfully!**")

    except Exception as e:
        st.error(f"⚠️ Error executing notebook: {e}")

    # ---------------------- APPLY CLUSTERING ----------------------
    st.write("🔄 **Applying clustering...**")

    if apply_clustering:
        df = apply_clustering(df, num_clusters, max_points_per_cluster)
    else:
        st.warning("⚠️ Using backup clustering method (KMeans).")
        df["Cluster"] = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit_predict(df[["Latitud", "Longitud"]].values)

    # ---------------------- ROUTE DISPLAY ----------------------
    agent_number = st.number_input("🔍 Select Agent", min_value=1, max_value=num_clusters, value=1)
    selected_cluster = agent_number - 1

    if "Cluster" in df.columns:
        cluster_data = df[df["Cluster"] == selected_cluster].copy()

        if cluster_data.shape[0] > 0:
            if tsp_nearest_neighbor:
                tsp_order = tsp_nearest_neighbor(cluster_data[["Latitud", "Longitud"]].values)
                if tsp_order is not None and len(tsp_order) > 0:
                    cluster_data = cluster_data.iloc[tsp_order]
                    cluster_data["Order"] = range(1, len(cluster_data) + 1)

                    st.write(f"### 📍 Optimized Route for Agent {agent_number}")
                    st.dataframe(cluster_data[["Order", "Nombre Comercial", "Latitud", "Longitud"]])

                    # ---------------------- EXPORT CSV ----------------------
                    buffer = io.BytesIO()
                    cluster_data.to_csv(buffer, index=False)
                    st.download_button(label="📥 Download Route as CSV", data=buffer.getvalue(), file_name=f"Route_Agent_{agent_number}.csv", mime="text/csv")

                    # ---------------------- DISPLAY MAP ----------------------
                    m = folium.Map(location=[cluster_data.iloc[0]["Latitud"], cluster_data.iloc[0]["Longitud"]], zoom_start=12)
                    for idx, row in cluster_data.iterrows():
                        folium.Marker(
                            location=[row["Latitud"], row["Longitud"]],
                            icon=folium.Icon(color="blue", icon="info-sign"),
                            popup=f"📌 {row['Nombre Comercial']}"
                        ).add_to(m)

                    st_folium(m, width=800, height=500)
                else:
                    st.error("⚠️ TSP function did not return a valid route.")
            else:
                st.error("⚠️ TSP function not found. Route optimization cannot be applied.")
        else:
            st.error(f"⚠️ No data available for Agent {agent_number}. Try a different one.")

    else:
        st.error("❌ Clustering failed. Please check your dataset and try again.")
