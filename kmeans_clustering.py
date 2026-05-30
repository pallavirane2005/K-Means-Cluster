# kmeans_clustering.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATASET
# ============================================
print("=" * 50)
print("📊 K-MEANS CUSTOMER SEGMENTATION")
print("=" * 50)

df = pd.read_csv('Mall_Customers.csv')
print(f"\n✅ Dataset loaded: {df.shape[0]} customers, {df.shape[1]} features\n")

# Display first few rows
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# ============================================
# 2. SELECT FEATURES FOR CLUSTERING
# ============================================
# We'll use Annual Income & Spending Score for 2D visualization
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Optional: Scale the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# 3. FIND OPTIMAL K USING ELBOW METHOD
# ============================================
print("\n" + "=" * 50)
print("🔍 Finding Optimal K (Elbow Method)")
print("=" * 50)

wcss = []  # Within-Cluster Sum of Squares
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.title('Elbow Method for Optimal K', fontsize=16, fontweight='bold')
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
print("📈 Elbow plot saved: elbow_method.png")
plt.show()

# ============================================
# 4. APPLY K-MEANS WITH OPTIMAL K=5
# ============================================
print("\n" + "=" * 50)
print("🎯 Applying K-Means (K=5)")
print("=" * 50)

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['Cluster'] = y_kmeans

print(f"\n✅ Clustering complete!")
print(f"Cluster distribution:")
print(df['Cluster'].value_counts().sort_index())

# ============================================
# 5. VISUALIZE CLUSTERS
# ============================================
print("\n" + "=" * 50)
print("📊 Generating Cluster Visualization")
print("=" * 50)

# Colors for each cluster
colors = ['red', 'blue', 'green', 'orange', 'purple']
cluster_names = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']

plt.figure(figsize=(12, 8))

# Plot each cluster
for i in range(optimal_k):
    plt.scatter(
        X[y_kmeans == i, 0], 
        X[y_kmeans == i, 1], 
        s=100, 
        c=colors[i], 
        label=f'Cluster {i}: {cluster_names[i]}',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

# Plot centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    s=300, 
    c='yellow', 
    marker='X', 
    label='Centroids',
    edgecolors='black',
    linewidth=2
)

plt.title('Customer Segments (K-Means Clustering)', fontsize=18, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=14)
plt.ylabel('Spending Score (1-100)', fontsize=14)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('customer_clusters.png', dpi=150, bbox_inches='tight')
print("📈 Cluster plot saved: customer_clusters.png")
plt.show()

# ============================================
# 6. ANALYZE EACH CLUSTER
# ============================================
print("\n" + "=" * 50)
print("📋 Cluster Analysis")
print("=" * 50)

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    print(f"\n🔹 Cluster {i} - {cluster_names[i]}")
    print(f"   Count: {len(cluster_data)} customers")
    print(f"   Avg Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
    print(f"   Avg Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}")
    print(f"   Avg Age: {cluster_data['Age'].mean():.1f}")

# ============================================
# 7. SAVE RESULTS
# ============================================
df.to_csv('Customer_Segments.csv', index=False)
print("\n" + "=" * 50)
print("✅ RESULTS SAVED")
print("=" * 50)
print("📁 Files created:")
print("   • elbow_method.png - Elbow curve")
print("   • customer_clusters.png - Cluster visualization")
print("   • Customer_Segments.csv - Data with cluster labels")
print("=" * 50)