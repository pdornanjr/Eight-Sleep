import data_pull
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df= data_pull.df

campaign_data=df.groupby(['date','campaign_name']).agg({ 'spend':'sum',
                                                        'transactions':'sum',
                                                        'visits':'sum',
                                                        'rev':'sum'
                                                        }).reset_index()
# campaign_data['cac']=campaign_data['spend']/campaign_data['transactions']

pd.set_option('display.max_rows', None)
campaign_data=campaign_data[campaign_data['spend']!=0]
# campaign_sorted=campaign_data.sort_values(by='spend', ascending=False)
# campaign_sorted.to_csv('campaign_sorted.csv')




# daily_cac=df.groupby(['date','week_number','year']).agg({'transactions':'sum','spend':'sum'}).reset_index()
# print(daily_cac)


cac_pivot_table = campaign_data.pivot_table(index=['campaign_name'],columns=['date'],values=[ 'spend',
                                                                                            'transactions',
                                                                                            'visits',
                                                                                               'rev'
                                                                                               ], aggfunc='sum')
cac_pivot_table=cac_pivot_table.fillna(0)
cac_pivot_table.columns = ['_'.join(map(str, col)).replace(' 00:00:00', '') for col in cac_pivot_table.columns]

print(cac_pivot_table)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cac_pivot_table)

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_data)

cac_pivot_table['cluster'] = kmeans.labels_

csv_info =cac_pivot_table[['cluster']]

csv_info.to_csv('only_paid_clusters.csv')




from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=reduced_data[:, 0],
    y=reduced_data[:, 1],
    hue=kmeans.labels_,
    palette='viridis',
    s=100
)
plt.title('K-means Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()
