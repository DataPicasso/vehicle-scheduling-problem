import streamlit as st

# Must be first!
st.set_page_config(page_title="🚀 Smart Route Optimization", layout="wide")

import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from sklearn.metrics import pairwise_distances
import googlemaps
import io
import os
import subprocess


# ---------------------- CUSTOM STYLING ----------------------
st.markdown("""
    <style>
        /* General Styles */
        body { font-family: 'Arial', sans-serif; background-color: #F5F5F5; }
        .stApp { background-color: white; }
        
        /* Title Styling */
        h1, h2, h3 { color: #002855; text-align: center; }
        h1 { font-size: 36px; }
        h2 { font-size: 28px; }
        h3 { font-size: 24px; }

        /* Sidebar Styling */
        .stSidebar { background-color: #ffffff; padding: 20px; }
        
        /* Buttons */
        .stButton>button { 
            background-color: #E63946; color: white; 
            border-radius: 10px; font-size: 16px; 
        }
        
        /* Sliders */
        .stSlider .st-ds { background-color: #457B9D !important; }
        
        /* Dataframe Styling */
        .dataframe { border-radius: 10px; overflow: hidden; }
    </style>
""", unsafe_allow_html=True)

# ---------------------- STREAMLIT APP SETUP ----------------------
st.set_page_config(page_title="🚀 Smart Route Optimization", layout="wide")

st.markdown("<h1>📍 Smart Route Optimization</h1>", unsafe_allow_html=True)
st.write("A minimalist & intelligent way to optimize travel routes with **Clustering & TSP**, powered by the **Vehicle Scheduling Problem (VSP)** model.")

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
        max_points_per_cluster = st.slider("🔹 Places per Agent", min_value=50, max_value=500, value=281)

    # ---------------------- HANDLE MISSING COORDINATES ----------------------
    if "Latitud" not in df.columns or "Longitud" not in df.columns:
        st.warning("Your dataset does not contain latitude and longitude. You must enter your **Google Maps API Key** to extract coordinates.")

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
            
            df["Latitud"], df["Longitud"] = zip(*df["Address"].apply(geocode_address))
            df.dropna(subset=["Latitud", "Longitud"], inplace=True)
            st.success("✅ Coordinates extracted successfully!")

    # ---------------------- EXECUTE NOTEBOOK ----------------------
    st.write("🔄 **Running the VSP Notebook...**")
    
    notebook_path = "/mnt/data/VSP.ipynb"

    if os.path.exists(notebook_path):
        try:
            subprocess.run(["jupyter", "nbconvert", "--to", "script", notebook_path], check=True)
            script_path = notebook_path.replace(".ipynb", ".py")

            with open(script_path, "r") as script_file:
                exec(script_file.read())

            st.success("✅ **Notebook executed successfully!**")
        except Exception as e:
            st.error(f"⚠️ Error executing the notebook: {e}")
    else:
        st.error("❌ Notebook file not found. Please upload the VSP notebook.")

    # ---------------------- ROUTE DISPLAY ----------------------
    agent_number = st.number_input("🔍 Select Agent", min_value=1, max_value=num_clusters, value=1)
    selected_cluster = agent_number - 1
    cluster_data = df[df["Cluster"] == selected_cluster].copy()

    if cluster_data.shape[0] > 0:
        # Apply TSP function from the executed notebook
        if "tsp_nearest_neighbor" in locals():
            tsp_order = tsp_nearest_neighbor(cluster_data[["Latitud", "Longitud"]].values)
            cluster_data = cluster_data.iloc[tsp_order]
            cluster_data["Order"] = range(1, len(cluster_data) + 1)

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
                    popup=f"Order {row['Order']}: {row['Nombre Comercial']}"
                ).add_to(m)

            # Show the map
            st_folium(m, width=800, height=500)

        else:
            st.error("❌ TSP function not found in the notebook. Ensure the notebook has a 'tsp_nearest_neighbor' function.")

    else:
        st.error(f"No data available for Agent {agent_number}. Try a different one.")
