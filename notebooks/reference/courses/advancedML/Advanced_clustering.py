import io
import matplotlib.pyplot as plt

import folium
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import MiniBatchKMeans, DBSCAN, OPTICS

from sklearn.metrics import silhouette_samples, silhouette_score, \
    calinski_harabasz_score, davies_bouldin_score


data_url = 'https://raw.githubusercontent.com/bahar/WorldCityLocations/master/World_Cities_Location_table.csv'

data_stream=requests.get(data_url).content

cols = ['id', 'country', 'city', 'lat', 'lon', 'population']
df=pd.read_csv(io.StringIO(data_stream.decode('utf-8')), sep=';', header=None)
df.columns = cols

# To make the map below faster to be plotted, we will only select 1000 random cities:
sample = df.sample(1000, random_state=100)

world_map = folium.Map(location=[0, 0], zoom_start=1 ,tiles = 'cartodbpositron', width=700, height=400)

for row in sample.iterrows():
    folium.CircleMarker(
        (row[1]['lat'], row[1]['lon']),
        radius=1,
        color='#0080bb',
        fill_color='#0080bb'
    ).add_to(world_map)

print('This is our raw data, plotted in a world map:')
print(world_map)

#Let's compare the performance of KMeans, DBSCAN and Optics clustering algorithms on the same dataset.
# A function to display the clustering results.
def plot_clustering_summary(clustering_dt, result):
    # We define this palette of colours to draw different clusters:
    palette = [
        '#ff6666', '#ffcc66', '#ccff66', '#66ff99', '#66e6ff', '#6666ff', '#e566ff',
        '#4da6ff', '#e60073', '#2200cc', '#0088cc', '#19ffff', '#1eb300', '#805500',
        '#7cb9e8', '#b0bf1a', '#5d8aa8', '#efdecd', '#3b7a57', '#967117', '#cce6ff',
        '#ffff99', '#ff0000', '#00ff00', '#0000ff', '#c7d9d6', '#d99100', '#1a0800'
    ]

    n_clusters = len(set(result))

    silhouette_avg = silhouette_score(clustering_dt, result)
    print("For n_clusters =", n_clusters)
    print("The average Silhouette score is :", round(silhouette_avg, 4))
    ch_score = calinski_harabasz_score(clustering_dt, result)
    print("The Calinski-Harabasz score is :", round(ch_score, 4))
    db_score = davies_bouldin_score(clustering_dt, result)
    print("The Davies-Bouldin score is :", round(db_score, 4))

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(clustering_dt, result)
    y_lower = 10
    fig, (ax1) = plt.subplots(figsize=(10, 7))
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(clustering_dt) + (n_clusters + 1) * 10])

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[result == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = palette[i % len(palette)]
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    result_map = folium.Map(location=[0, 0], zoom_start=1, tiles='cartodbpositron',
                            width=700, height=400)
    for i in range(len(clustering_dt)):
        folium.CircleMarker(
            (clustering_dt.iloc[i]['lat'], clustering_dt.iloc[i]['lon']),
            radius=1,
            color=palette[result[i] % len(palette)],
            fill_color=palette[result[i] % len(palette)]
        ).add_to(result_map)

    plt.show()
    return result_map

# We will cluster data by their latitude and longitude (no city names or countries)
clustering_data = sample[['lat', 'lon']]

# Change this value to modify the results from K-means.
# For example... there are 6 continents... That's some a-priori knowledge
NUM_CLUSTERS = 10

# We will leave the other parameters not here as 'default'
kmeans = MiniBatchKMeans(
    n_clusters=NUM_CLUSTERS,
    init='k-means++',
)
kmeans_clusters = kmeans.fit_predict(clustering_data)

res_map = plot_clustering_summary(clustering_data, kmeans_clusters)
print(res_map)

#DBSCAN
dbscan = DBSCAN(
    eps=6,
    min_samples=5
)
dbscan_clusters = dbscan.fit_predict(clustering_data)

res_map = plot_clustering_summary(clustering_data, dbscan_clusters)
print(res_map)

#OPTICS
optics = OPTICS(
    min_samples=20
)
optics_clusters = optics.fit_predict(clustering_data)

res_map = plot_clustering_summary(clustering_data, optics_clusters)
print(res_map)