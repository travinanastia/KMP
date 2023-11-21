from tkinter import N
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tabulate import tabulate 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

path = "C:\\Users\\N\\Documents\\KMP\\Yeast.csv"
dataset = pd.read_csv(path)
missing = dataset.isna()
dataset.iloc[:, :-1] = dataset.iloc[:, :-1].apply(lambda col: col.fillna(col.mean()), axis=0) 
dataset.drop_duplicates(inplace=True)

#Task 2-----------------------------------------------------------------
data = dataset.iloc[:,:-1].values
clas = dataset.iloc[:,-1].values

#scaler = MinMaxScaler()
#scaler_data = scaler.fit_transform(data)
scaler = StandardScaler()
scaler.fit(data)
scaler_data = scaler.transform(data)

clusterer = KMeans(n_clusters=10)
clusterer.fit(scaler_data)
labels = clusterer.labels_
metrics.silhouette_score(scaler_data, labels, metric='euclidean')
predictions = clusterer.predict(scaler_data)
dataset["cluster"] = predictions
print("\nРезультати кластеризації даних за алгоритмом k-means\n")
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
print(dataset, "\n")


#Task 4-----------------------------------------------------------------
count_cluster = Counter(labels)
print("Кількість об'єктів у кластерах:")
for cluster, count in count_cluster.items():
    print(f"Кластер {cluster}: {count} об'єктів")
print()

print("\nКількість об'єктів класу у кожному кластері:")
cluster_content = dataset.groupby(["cluster", "name"]).size().unstack(fill_value=0)
cluster_content['Total'] = cluster_content.sum(axis=1)
cluster_content.loc['Total'] = cluster_content.sum()
print(tabulate(cluster_content, headers="keys", tablefmt="psql"))

#Task 3-----------------------------------------------------------------
centroids = clusterer.cluster_centers_

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Результати кластеризації та центроїди")

scatter1 = axes[0, 0].scatter(scaler_data[:, 0], scaler_data[:, 1], c=predictions, s=15, cmap="brg")
handles, labels = scatter1.legend_elements()
legend1 = axes[0, 0].legend(handles, labels, loc="upper right")
axes[0, 0].add_artist(legend1)
scatter2 = axes[0, 0].scatter(centroids[:, 0], centroids[:, 1], c="black", s=100, marker="x", linewidths=2, label="centroids")
axes[0, 0].legend(loc="upper left")
axes[0, 0].set_xlabel(f"{dataset.columns[0]}")
axes[0, 0].set_ylabel(f"{dataset.columns[1]}")

scatter1 = axes[0, 1].scatter(scaler_data[:, 4], scaler_data[:, 5], c=predictions, s=15, cmap="brg")
handles, labels = scatter1.legend_elements()
legend1 = axes[0, 1].legend(handles, labels, loc="upper right")
axes[0, 1].add_artist(legend1)
scatter2 = axes[0, 1].scatter(centroids[:, 4], centroids[:, 5], c="black", s=100, marker="x", linewidths=2, label="centroids")
axes[0, 1].legend(loc="upper left")
axes[0, 1].set_xlabel(f"{dataset.columns[4]}")
axes[0, 1].set_ylabel(f"{dataset.columns[5]}")

scatter1 = axes[1, 0].scatter(scaler_data[:, 2], scaler_data[:, 4], c=predictions, s=15, cmap="brg")
handles, labels = scatter1.legend_elements()
legend1 = axes[1, 0].legend(handles, labels, loc="upper right")
axes[1, 0].add_artist(legend1)
scatter2 = axes[1, 0].scatter(centroids[:, 2], centroids[:, 4], c="black", s=100, marker="x", linewidths=2, label="centroids")
axes[1, 0].legend(loc="upper left")
axes[1, 0].set_xlabel(f"{dataset.columns[2]}")
axes[1, 0].set_ylabel(f"{dataset.columns[4]}")

scatter1 = axes[1, 1].scatter(scaler_data[:, 0], scaler_data[:, 7], c=predictions, s=15, cmap="brg")
handles, labels = scatter1.legend_elements()
legend1 = axes[1, 1].legend(handles, labels, loc="upper right")
axes[1, 1].add_artist(legend1)
scatter2 = axes[1, 1].scatter(centroids[:, 0], centroids[:, 7], c="black", s=100, marker="x", linewidths=2, label="centroids")
axes[1, 1].legend(loc="upper left")
axes[1, 1].set_xlabel(f"{dataset.columns[0]}")
axes[1, 1].set_ylabel(f"{dataset.columns[7]}")
plt.show()

#Task 5-----------------------------------------------------------------
print("\nВизначення оптимальної кількості кластерів")
df = pd.DataFrame(columns=["К-ть кластерів", "СКВ", "Силует", "ДБ"])
for i in range(2,15):
    clusterer_i= KMeans(n_clusters=i).fit(scaler_data)
    predictions_i=clusterer_i.predict(scaler_data)

    WCSS = clusterer_i.inertia_

    Silhouette = metrics.silhouette_score(scaler_data,predictions_i)

    DB = metrics.davies_bouldin_score(scaler_data, predictions_i)

    new_row_df = pd.DataFrame([[i,WCSS,Silhouette,DB]], columns = df.columns)
    df = pd.concat([df,new_row_df], ignore_index = True)

print(tabulate(df, headers = "keys", tablefmt = "psql", floatfmt=".3f"))


plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(df["К-ть кластерів"], df["СКВ"], marker="o", color="maroon",linestyle="None", label="СКВ")
plt.xlabel("К-ть кластерів")
plt.ylabel("СКВ")
plt.title("Метод ліктя")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df["К-ть кластерів"], df["Силует"], marker="o", color="maroon",linestyle="None", label="Силует")
plt.xlabel("К-ть кластерів")
plt.ylabel("Силует")
plt.title("Метод силуету")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df["К-ть кластерів"], df["ДБ"], marker="o",color="maroon", linestyle="None", label="ДБ")
plt.xlabel("К-ть кластерів")
plt.ylabel("ДБ")
plt.title("Індекс Девіса-Булдіна")
plt.legend()
plt.tight_layout() 
plt.show()