import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Meminta pengguna memasukkan 5 nilai elemen array dalam satu dimensi
arr = []
n = int(input("Masukan Jumlah Baris : "))
for i in range(n):
    row = []
    for j in range(2) :
        val = float(input("Masukkan nilai elemen baris {} kolom {}: ".format(i+1, j+1)))
        row.append(val)
    arr.append(row)

# Mengonversi nilai-nilai tersebut menjadi np array
data = np.array(arr)

# Menampilkan hasil array yang telah dibuat
print("Array yang telah dibuat: ", data)


# # Generate data
# data = np.array([[1, 3], [3, 3], [4, 3], [5, 3], [1, 2], [4, 2], [1, 1], [2, 1],[2,6],[3,8]])

# Visualize the data
plt.scatter(data[:, 0], data[:, 1])
plt.show()

# Create KMeans object and fit the data
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# Print the centroids
centroids = kmeans.cluster_centers_
print("Centroids:", centroids)

# Visualize the clusters
colors = ["g.", "r.","b."]
for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], colors[kmeans.labels_[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()
