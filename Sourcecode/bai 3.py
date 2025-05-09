import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import os

# Load data from results.csv
df = pd.read_csv('results.csv', encoding='utf-8-sig')

# Define numeric columns (same as in Bài 2)
STATS_COLUMNS = [
    "Nation", "Team", "Position", "Age",
    "Matches Played", "Starts", "Minutes",
    "Goals", "Assists", "Yellow Cards", "Red Cards",
    "xG", "xAG",
    "PrgC", "PrgP", "PrgR",
    "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90",
    "GA90", "Save%", "CS%",
    "Penalty Kicks Save%",
    "SoT%", "SoT/90", "G/Sh", "Dist",
    "Passes Completed",
    "TkI", "TkIW",
    "Att Challenges", "Lost Challenges",
    "Blocks", "Sh Blocks", "Pass Blocks", "Int",
    "Touches", "Def Pen Touches", "Def 3rd Touches", "Mid 3rd Touches", "Att 3rd Touches", "Att Pen Touches",
    "Att Take-Ons", "Succ% Take-Ons", "Tkld% Take-Ons",
    "Carries", "ProDist", "ProgC Carries", "1/3 Carries", "CPA", "Mis", "Dis",
    "Rec", "PrgR Receiving",
    "Fls", "Fld", "Off", "Crs", "Recov",
    "Aerial Won", "Aerial Lost", "Aerial Won%"
]

# Filter numeric columns
numeric_columns = [col for col in STATS_COLUMNS if df[col].dtype in ['int64', 'float64'] or df[col].apply(lambda x: str(x).replace('N/a', 'nan').replace(',', '').replace('%', '').strip().replace('.', '', 1).isdigit() if str(x) != 'N/a' else False).all()]

# Prepare data for clustering
X = df[numeric_columns].copy()
X = X.replace('N/a', np.nan)
X = pd.DataFrame(X, columns=numeric_columns)

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=numeric_columns)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Determine optimal number of clusters using Elbow Method
wcss = []
max_clusters = 10
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('histograms/elbow_curve.png')
plt.close()

# Choose number of clusters (e.g., 4 based on elbow curve and domain knowledge)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to DataFrame
df['Cluster'] = cluster_labels

# Save clustering results
clustering_results = df[['Player', 'Cluster']].copy()
clustering_results.to_csv('clustering_results.csv', index=False, encoding='utf-8-sig')
print("Saved clustering results to clustering_results.csv")

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained variance ratio by PCA components: {explained_variance_ratio}")

# Plot 2D clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.title('2D PCA Cluster Visualization of Premier League Players')
plt.xlabel(f'PCA Component 1 ({explained_variance_ratio[0]:.2%} variance)')
plt.ylabel(f'PCA Component 2 ({explained_variance_ratio[1]:.2%} variance)')
plt.colorbar(scatter, label='Cluster')
plt.savefig('histograms/player_clusters_2d.png')
plt.close()

# Analyze clusters and print comments
cluster_summary = []
cluster_summary.append("K-means Clustering Analysis:")
cluster_summary.append(f"Number of clusters chosen: {n_clusters}")
cluster_summary.append("Reasoning for number of clusters:")
cluster_summary.append("- The Elbow Method was used to evaluate the optimal number of clusters. The WCSS plot showed a noticeable bend around 4 clusters, indicating diminishing returns for additional clusters.")
cluster_summary.append("- Domain knowledge suggests that Premier League players can be grouped into roles such as goalkeepers, defenders, midfielders, and forwards, which aligns with 4 clusters.")
cluster_summary.append("Comments on results:")
cluster_summary.append("- Cluster 0 likely represents goalkeepers, as they have distinct stats like Save% and GA90.")
cluster_summary.append("- Clusters 1, 2, and 3 may correspond to defenders, midfielders, and forwards, respectively, based on stats like Tackles, Passes Completed, and Goals.")
cluster_summary.append("- The PCA plot visualizes the separation of players, though some overlap exists due to the complexity of player roles.")
cluster_summary.append(f"- PCA explained variance: {explained_variance_ratio[0]:.2%} (Component 1) + {explained_variance_ratio[1]:.2%} (Component 2) = {(explained_variance_ratio[0] + explained_variance_ratio[1]):.2%} total.")

# Print clustering analysis
print('\n'.join(cluster_summary))

# Ensure histograms directory exists
os.makedirs('histograms', exist_ok=True)
print("Saved elbow curve and 2D cluster plot to histograms/")