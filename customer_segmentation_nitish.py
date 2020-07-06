import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv("Wholesale_customers_data.csv")
dataset.drop(['Region', 'Channel'], axis = 1, inplace = True)


data_copy = dataset.copy()
grocery_data = data_copy['Grocery']
data_copy.drop(['Grocery'], axis=1, inplace = True)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_copy, grocery_data, test_size = 0.2, random_state=0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
score = regressor.score(X_test, y_test)


plt.style.use('seaborn-deep')
plt.hist([y_pred, y_test], bins='auto', label=['y_pred', 'y_test'])
plt.legend(loc='upper right')
plt.title("Predicted sales vs Actual sales")
plt.xlabel("Average sales")
plt.ylabel("Average frequency")


correlations = dataset.corr(method='pearson')
ax = plt.axes()
sns.heatmap(correlations, vmin=-1, vmax=1, annot=True)
ax.set_title('Correlation among attributes')


pd.plotting.scatter_matrix(dataset, figsize = (14,8), diagonal = 'kde');


plt.plot(pd.DataFrame.skew(dataset), label='Actual value')
plt.plot([1,1,1,1,1,1], label='Expected value')
plt.title("Skewness of attributes")
plt.legend(loc='upper right')
plt.yticks(range(0,15,1))
plt.xlabel("Attributes")
plt.ylabel("Skewness value")


log_data = dataset.apply(np.log)
pd.plotting.scatter_matrix(log_data, figsize = (14,8), diagonal = 'kde');


import visuals as  vs
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
pca.fit(log_data)
pca_results = vs.pca_results(log_data, pca)
pca_cum_results=pca_results.cumsum()


ax = plt.axes()
sns.heatmap(pca_results, vmin=-1, vmax=1, annot=True)
ax.set_title('Explained varience at various dimentions')


ax = plt.axes()
sns.heatmap(pca_cum_results, vmin=-1, vmax=1, annot=True)
ax.set_title('Cumulative explained varience at various dimentions')


pca = PCA(n_components=2)
pca.fit(log_data)
reduced_data = pca.transform(log_data)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
vs.biplot(log_data, reduced_data, pca)


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(reduced_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters = 2,init = 'k-means++')
y_kmeans = kmeans.fit_predict(reduced_data)
reduced_data['Clusters']=kmeans.labels_
centers = np.array(kmeans.cluster_centers_)


plt.scatter(reduced_data.loc[reduced_data['Clusters'] == 0] ['Dimension 1'],
            reduced_data.loc[reduced_data['Clusters'] == 0] ['Dimension 2'],
            c='blue',s=5)
plt.scatter(reduced_data.loc[reduced_data['Clusters'] == 1] ['Dimension 1'],
            reduced_data.loc[reduced_data['Clusters'] == 1] ['Dimension 2'],
            c='red',s=5)
plt.scatter(reduced_data.loc[reduced_data['Clusters'] == 2] ['Dimension 1'],
            reduced_data.loc[reduced_data['Clusters'] == 2] ['Dimension 2'],
            c='orange',s=5)
plt.scatter(centers[:,0], centers[:,1], marker="x", color='yellow')
plt.title('Customer Segmetation') 
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')