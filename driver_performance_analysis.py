import base64
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def show_driver_performance_analysis():
    background_image_path = "download.png" 
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url('data:image/jpg;base64,{base64.b64encode(open(background_image_path, "rb").read()).decode()}');
                background-size: cover;
                background-position: center center;
                background-repeat: no-repeat;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title of the app
    st.title('Formula 1 Driver Performance Analysis')

    # Load CSV files (assuming they are in the current directory)
    drivers_file = 'f1db-drivers.csv'
    results_file = 'f1db-races-race-results.csv'
    races_file = 'f1db-races.csv'

    # Load the data
    drivers = pd.read_csv(drivers_file)
    results = pd.read_csv(results_file)
    races = pd.read_csv(races_file)

    # Merge results with race data to filter by year (from 2013 onward)
    results_with_year = results.merge(races[['id', 'year']], left_on="raceId", right_on="id")
    results_2013_onward = results_with_year[results_with_year['year_x'] >= 2024]

    # Merge with driver data
    merged = results_2013_onward.merge(drivers, left_on="driverId", right_on="id")

    # Group by driver and compute performance metrics
    performance = merged.groupby("fullName").agg(
        races_participated=('raceId', 'count'),
        total_points=('points', 'sum')
    ).reset_index()

    # Compute avg points per race
    performance['avg_points_per_race'] = performance['total_points'] / performance['races_participated']

    # Filter drivers with meaningful participation
    performance = performance[performance['races_participated'] >= 1]

    # Prepare data for clustering
    features = performance[['total_points', 'avg_points_per_race', 'races_participated']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    performance['cluster'] = kmeans.fit_predict(scaled_features)

    # Rank clusters by average total_points and relabel
    cluster_means = performance.groupby('cluster')['total_points'].mean().sort_values(ascending=False)
    cluster_order = {old: new for new, old in enumerate(cluster_means.index)}
    performance['cluster_ranked'] = performance['cluster'].map(cluster_order)

    # Add human-readable cluster labels
    cluster_labels = {
        0: "Elite Performers",
        1: "Strong Competitors",
        2: "Developing Drivers",
        3: "Low Impact Drivers"
    }
    performance['cluster_label'] = performance['cluster_ranked'].map(cluster_labels)

    # Use PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_features)
    performance['PC1'] = pca_components[:, 0]
    performance['PC2'] = pca_components[:, 1]

    # Dropdown for selecting Driver Name
    driver_name_dropdown = st.selectbox("Select a Driver to Find Their Cluster:", performance['fullName'].unique())

    if driver_name_dropdown:
        # Find the selected driver's information
        driver_info = performance[performance['fullName'] == driver_name_dropdown].iloc[0]
        
        # Display the driver's cluster and performance data
        st.subheader(f"{driver_info['fullName']} belongs to the cluster:")
        st.write(f"Cluster: {driver_info['cluster_label']}")
        st.write(f"Total Points: {driver_info['total_points']}")
        st.write(f"Avg Points per Race: {driver_info['avg_points_per_race']:.2f}")
        st.write(f"Races Participated: {driver_info['races_participated']}")

        # Create a scatter plot for PCA results, highlighting the selected driver
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot all drivers
        ax.scatter(performance['PC1'], performance['PC2'], c=performance['cluster'], cmap='viridis', label='All Drivers', alpha=0.6)

        # Highlight the selected driver
        selected_driver = performance[performance['fullName'] == driver_name_dropdown]
        ax.scatter(selected_driver['PC1'], selected_driver['PC2'], color='red', label=driver_name_dropdown, s=100, edgecolor='black', zorder=5)

        ax.set_title(f"PCA Plot - {driver_name_dropdown} Performance Cluster")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend()
        ax.grid(True)

        # Display the plot in Streamlit
        st.pyplot(fig)
