import streamlit as st
import pandas as pd
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import pickle

@st.cache_data
def load_and_process_data():
    with open("processed_taxi_data.pkl", "rb") as f:
        data = pickle.load(f)
        df = pd.DataFrame(data['all_points'], 
                         columns=['longitude', 'latitude', 'datetime'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

st.title("Taxi Data Visualization")
points_df = load_and_process_data()

st.sidebar.header("Time Filter")
start_hour, end_hour = st.sidebar.slider("Time Range (24h)", 0, 23, (0, 23))

time_mask = (points_df['datetime'].dt.hour >= start_hour) & \
           (points_df['datetime'].dt.hour <= end_hour)
filtered_df = points_df[time_mask]

st.sidebar.header("Visualization Settings")
view_type = st.sidebar.radio("View", 
    ["Beijing Overview", "Inner City (5th Ring Road)"])

extent = [116.2, 116.55, 39.8, 40.05] if view_type == "Inner City (5th Ring Road)" \
    else [116.1, 116.8, 39.6, 40.2]
title = "Within the 5th Ring Road of Beijing" if view_type == "Inner City (5th Ring Road)" \
    else "Data Overview in Beijing"

region_mask = (filtered_df['longitude'] >= extent[0]) & \
             (filtered_df['longitude'] <= extent[1]) & \
             (filtered_df['latitude'] >= extent[2]) & \
             (filtered_df['latitude'] <= extent[3])
region_filtered_df = filtered_df[region_mask]

if not region_filtered_df.empty:
    canvas = ds.Canvas(plot_width=800, plot_height=800,
                      x_range=(extent[0], extent[1]),
                      y_range=(extent[2], extent[3]))
    
    fire_colors = plt.cm.colors.ListedColormap(cc.fire)
    agg = canvas.points(region_filtered_df, 'longitude', 'latitude')
    img = tf.shade(agg, how='log', cmap=cc.fire)
    
    x_lines = np.linspace(extent[0], extent[1], 10)
    y_lines = np.linspace(extent[2], extent[3], 10)
    
    for x in x_lines:
        line_df = pd.DataFrame({'x': [x, x], 'y': [extent[2], extent[3]]})
        line_agg = canvas.line(line_df, 'x', 'y')
        line_img = tf.shade(line_agg, cmap=['white'], alpha=0.3)
        img = tf.stack(img, tf.set_background(line_img))
    
    for y in y_lines:
        line_df = pd.DataFrame({'x': [extent[0], extent[1]], 'y': [y, y]})
        line_agg = canvas.line(line_df, 'x', 'y')
        line_img = tf.shade(line_agg, cmap=['white'], alpha=0.3)
        img = tf.stack(img, tf.set_background(line_img))
    
    img = tf.set_background(img, 'black')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    ax.imshow(img.to_pil(), extent=extent)
    ax.set_aspect((extent[1] - extent[0]) / (extent[3] - extent[2]))
    
    norm = LogNorm(vmin=1, vmax=agg.values.max())
    sm = ScalarMappable(norm=norm, cmap=fire_colors)
    cb = plt.colorbar(sm, ax=ax)
    cb.set_label('Point Density (log scale)', color='white', size=10)
    cb.ax.tick_params(colors='white')
    
    ax.set_xlabel('Longitude', color='white', size=10)
    ax.set_ylabel('Latitude', color='white', size=10)
    ax.set_title(f"{title}\n{start_hour:02d}:00-{end_hour:02d}:00", 
                 color='white', pad=20, size=12)
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('white')
    
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    st.pyplot(fig)
    
    # Sidebar stats
    with st.sidebar:
        st.markdown("### Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Points", f"{len(region_filtered_df):,}")
        with col2:
            st.metric("Time Range", f"{start_hour:02d}:00-{end_hour:02d}:00")
else:
    st.warning("No data points in selected range")