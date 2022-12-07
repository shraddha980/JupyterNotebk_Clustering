#!/usr/bin/env python
# coding: utf-8

# ### Unsupervised Learning

# * We devide our dataset into supervised or unsupervised learning depending on the fact that if labelled information is given
# to us or not and if we want to make use of that labelled information to make conclusions.Here, we do not have any prior 
# information about the dataset. We are supposed to find patterns in our dataset using given information.Such a problem statement falls under unsupervised learning. There are different types of methods used to detect patterns in the dataset like 
# centroid based clustering or non hierarchical based clustering/ Distance based clustering which consists of K-means, K-means++, Density based clustering which consists DBSCAN Clustering,Distribution based clustering, Hierarchical Clustering which consists of Aglomerative Clustering.
# 
# * Clustering algorithms have multiple applications like customer segmentation, anomly detection, image segmentation, semi-supervised learning, for dimensionality reduction, data analysis etc.

# ### Exploratory Data Analysis

# In[153]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px


# In[5]:


#loading the data
df = pd.read_csv(r"https://raw.githubusercontent.com/NelakurthiSudheer/Mall-Customers-Segmentation/main/Dataset/Mall_Customers.csv")


# ### DataSet Overview
> The Data Set we use is the Mall Customers Dataset
> It consists of 5 columns
> Each customer has a unique customer ID
> The Gender Column tells us if customer is Male or Female
> The Age Column gives us the approximate age of the customer
> The Annual Income gives us the income of customer in Thousand Dollars
> The Spending Score tells us about spending habit of customer of 100 being the highest score. It is given by the mall to the
customers for their spending behavior
# ### Business Problem
Based on the above 4 Columns we have to group customers having a high spending score ( The most Loyal Customers ).
# ### STATISTICAL ANALYSIS

# In[6]:


#checking the first 10 rows of data
df.head(10)


# In[7]:


#checking statistics of data
df.describe()


# In[20]:


#check no of unique values in each column
df.nunique()


# In[22]:


#check no of rows and columns 
df.shape


# In[25]:


#check null values
df.isnull().sum()

Observation:

> We have customers between 18 to 70 yrs of age group. The age of most customers is arounf 36yrs ( median age group )
> We have 75% of customers euqal to or below 49yrs of age.
> We have customers whose annual income is between 15k to 137k.
> We have 75% of customers with income equal to or below 78k
> We have 0 Null Values
> We have Male and Female Customers
> We have 200 Rows and 5 Columns
> Age, Annual Income and Spending Score have multiple Unique Values
# #### GRAPHICAL ANALYSIS

# In[33]:


#check count of male Vs females
import seaborn as sns
sns.countplot(df['Gender'])


# The Mall has more number of Female Customers

# In[96]:


# Age Distribution plot
sns.histplot(data=df, x="Age", bins = list(range(10, 100, 10)))
plt.title("Distribution of Customer's Age (Yrs)")


# There are more customers <45 years of age. The Distribution of Age variable is Right-Skewed.

# In[45]:


# Annual Income Distribution plot
sns.histplot(data=df, x="Annual Income (k$)", bins = list(range(10, 200, 10)))
plt.title("Distribution of Customer's Annual Income(k$)")


# The Distribution of Customers Annual Income variable is also Right-Skewed. Most customers have income in the range 50k - 80k.

# In[50]:


# Spending Dist Score Distribution plot
sns.histplot(data=df, x=" ", bins = list(range(10, 130, 10)))
plt.title("Distribution of Customer's Spending Score (1-100)")


# Spending Score Variable has normal distribution. Both high-spending and low-spending is equally spread at both the ends.

# In[174]:


# Relationship between Age and Annual Income

sns.scatterplot(data=df, x="Annual Income (k$)",
                 y="Age"
                )


# In[85]:


# Relationship between Gender and Annual Income

sns.scatterplot(data=df, x="Gender",
                 y="Annual Income (k$)")


# In[83]:


# Relationship between Age and Spending Score

sns.scatterplot(data=df, x="Age",
                 y="Spending Score (1-100)")


# In[82]:


# Relationship between Gender and Spending Score

sns.scatterplot(data=df, x="Gender",
                 y="Spending Score (1-100)")
                


# In[68]:


df.boxplot(column=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])


# #### REMOVING OUTLIERS

# In[87]:


# Importing
import sklearn

''' Detection '''
# IQR
Q1 = np.percentile(df['Annual Income (k$)'], 25,interpolation = 'midpoint')

Q3 = np.percentile(df['Annual Income (k$)'], 75,interpolation = 'midpoint')
IQR = Q3 - Q1

print("Old Shape: ", df.shape)

# Upper bound
upper = np.where(df['Annual Income (k$)'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(df['Annual Income (k$)'] <= (Q1-1.5*IQR))

''' Removing the Outliers '''
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)

print("New Shape: ", df)


# In[88]:


df.shape


# #### CATEGORICAL TO NUMERICAL CONVERSION

# In[100]:


#Converting gender column to numerical column
df = pd.get_dummies(df, columns=['Gender'])


# In[109]:


#Dropping CustomerID Column
df.drop(['CustomerID'], axis=1)


# #### STANDARD SCALING 

# In[114]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
std_scaler = StandardScaler()
df_scaled = std_scaler.fit_transform(df)


# In[115]:


df_scaled


# ### Non-Hierarchical / Distance/ Centroid Based Clustering
# #### K-MEANS IMPLEMENTATION

# In[182]:


# Finding the optimal No of Clusters for K-Means implementation using WCSS (inertia method)


# In[122]:


error = []


# The Elbow method runs k-means clustering on dataset for a range of values between 1 to 11. Then for each value of k computes an average score for all clusters. 

# In[123]:



for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df_scaled)
    error.append(model.inertia_)


# In[132]:


plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('Error of Cluster')
sns.pointplot(x=list(range(1, 11)), y=error)


# In[175]:


error_kmeans = []


# In[176]:


for k in range(1, 11):
    model = KMeans(n_clusters=k, init = 'k-means++', random_state=42)
    model.fit(df_scaled)
    error_kmeans.append(model.inertia_)


# In[177]:


plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('Error of Cluster')
sns.pointplot(x=list(range(1, 11)), y=error_kmeans)


# #### Apply K=3,4,5,6 and checking Silhoutte coffiecient respectively

# In[255]:


model = KMeans(n_clusters = 3, random_state=42)
model.fit(df_scaled)


# In[263]:


model.labels_


# In[256]:


df = df.assign(ClusterLabel = model.labels_)


# In[257]:


df.groupby("ClusterLabel")[["Age","Annual Income (k$)","Spending Score (1-100)","Gender_Female","Gender_Male"]].mean()


# In[265]:


from sklearn.metrics import silhouette_samples, silhouette_score,calinski_harabasz_score,davies_bouldin_score

print("Silhouette Coefficient: %0.3f" % silhouette_score(df_scaled, model.labels_))
print("Calinski-Harabasz Index: %0.3f" % calinski_harabasz_score(df_scaled, model.labels_))
print("Davies-Bouldin Index: %0.3f" % davies_bouldin_score(df_scaled, model.labels_))
   


# In[266]:


model = KMeans(n_clusters = 4, random_state=42)
model.fit(df_scaled)


# In[267]:


df = df.assign(ClusterLabel = model.labels_)


# In[268]:


df.groupby("ClusterLabel")[["Age","Annual Income (k$)","Spending Score (1-100)","Gender_Female","Gender_Male"]].mean()


# In[269]:


from sklearn.metrics import silhouette_samples, silhouette_score,calinski_harabasz_score,davies_bouldin_score

print("Silhouette Coefficient: %0.3f" % silhouette_score(df_scaled, model.labels_))
print("Calinski-Harabasz Index: %0.3f" % calinski_harabasz_score(df_scaled, model.labels_))
print("Davies-Bouldin Index: %0.3f" % davies_bouldin_score(df_scaled, model.labels_))
   


# In[270]:


model = KMeans(n_clusters = 5, random_state=42)
model.fit(df_scaled)


# In[271]:


df = df.assign(ClusterLabel = model.labels_)


# In[272]:


df.groupby("ClusterLabel")[["Age","Annual Income (k$)","Spending Score (1-100)","Gender_Female","Gender_Male"]].mean()


# In[273]:


from sklearn.metrics import silhouette_samples, silhouette_score,calinski_harabasz_score,davies_bouldin_score

print("Silhouette Coefficient: %0.3f" % silhouette_score(df_scaled, model.labels_))
print("Calinski-Harabasz Index: %0.3f" % calinski_harabasz_score(df_scaled, model.labels_))
print("Davies-Bouldin Index: %0.3f" % davies_bouldin_score(df_scaled, model.labels_))
   


# In[274]:


model = KMeans(n_clusters = 6, random_state=42)
model.fit(df_scaled)


# In[275]:


df = df.assign(ClusterLabel = model.labels_)


# In[276]:


df.groupby("ClusterLabel")[["Age","Annual Income (k$)","Spending Score (1-100)","Gender_Female","Gender_Male"]].mean().round()


# In[277]:


from sklearn.metrics import silhouette_samples, silhouette_score,calinski_harabasz_score,davies_bouldin_score

print("Silhouette Coefficient: %0.3f" % silhouette_score(df_scaled, model.labels_))
print("Calinski-Harabasz Index: %0.3f" % calinski_harabasz_score(df_scaled, model.labels_))
print("Davies-Bouldin Index: %0.3f" % davies_bouldin_score(df_scaled, model.labels_))
   


# In[280]:


facet = sns.lmplot(data=df, x='Age', y='Annual Income (k$)', hue='ClusterLabel', 
                   fit_reg=False, legend=True, legend_out=True)


# In[281]:


facet = sns.lmplot(data=df, x='Age', y='Spending Score (1-100)', hue='ClusterLabel', 
                   fit_reg=False, legend=True, legend_out=True)


# From above plot we can se that our model has not separated clusters efficiently. The same can is visible in Silhouette Coefficient

# # DBSCAN Clustering 
#    Density Based Spatial Clsutering of Applications

# DBSCAN separates the high density regions of the data from low-density areas. The no of clusters in prior do not need to be provided for this algorithm. It consides clusters as continous regions of high density. This helps us track outliers as the points in the low density regions. Hence, it is not sensitive to Outliers as K-means Clustering.
# 
# DBSCAN PARAMETERS
# 
# Epsilon : The minimum distance between two points to be termed as neighbours. 
# MinPoints : This refers to the minimum number of points needed to construct a cluster.
# 
# DATA POINTS in DBSCAN
# 
# Core: This is a point from which the two parameters above are fully defined, i.e., a point with at least Minpoints within the Eps distance from itself.
# 
# Border: This is any data point that is not a core point, but it has at least one Core point within Eps distance from itself.
# 
# Noise: This is a point with less than Minpoints within distance Eps from itself. Thus, it’s not a Core or a Border.
# 
# 

# In[379]:


df1 = pd.read_csv(r"https://raw.githubusercontent.com/NelakurthiSudheer/Mall-Customers-Segmentation/main/Dataset/Mall_Customers.csv")


# In[380]:


df1.drop(['CustomerID'], axis=1, inplace=True)


# In[381]:


df1 = pd.get_dummies(df1, columns=['Gender'])


# In[482]:


from sklearn.cluster import DBSCAN
import numpy as np


# In[487]:


DBS_clustering = DBSCAN(eps=12.5, min_samples=4).fit(df1)

DBSCAN_clustered = df1.copy()
DBSCAN_clustered.loc[:,'Cluster'] = DBS_clustering.labels_ 


# In[488]:


y_pred = dbscan.fit_predict(df1)


# In[489]:


y_pred


# In[490]:


df1 = df1.assign(ClusterLabel = dbscan.labels_)


# In[491]:


df1.groupby("ClusterLabel")[["Age","Annual Income (k$)","Spending Score (1-100)","Gender_Female","Gender_Male"]].mean().round()


# In[492]:


plt.figure(figsize=(10,6))
plt.scatter(x=df1['Age'], y=df1['Spending Score (1-100)'],c=y_pred, cmap='Paired')
plt.title("Clusters determined by DBSCAN")


# In[494]:


DBSCAN_clust_sizes = DBSCAN_clustered.groupby('Cluster').size().to_frame()
DBSCAN_clust_sizes.columns = ["DBSCAN_size"]
DBSCAN_clust_sizes


# # Non-Hierarchical Clustering

# Non-Hierarchical Clustering can be divided into Bottom-Up ( Aglomerative ) and Top -Down ( Decisive ) Clustering. This type of clustering does not need to enter the no of K. Instead it creates a dendogram or Tree Based Structure.

# #### Aglomerative Clustering

# In[244]:


from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


# Linkage Methods
# 
# Single Linkage: Minimum Distance between instances of two considered clusters. The distance calculation can be either Euclidean or Manhattan Distance
# 
# Complete Linkage: The farthest distance between the instances of two class. 
# 
# Average Linkage: The average distance between all linkages is considered.
# 
# Centroid Linkage: The distance between the two centroid is considered.
# 
# Ward Linkage: The distance between two clusters say A and B are how much the sum of squares will increase when they both are merged. 
# 

# In[510]:


plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram_ward = shc.dendrogram((shc.linkage(df_scaled,metric='euclidean', method ='ward')))


# In[511]:


plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram_average = shc.dendrogram((shc.linkage(df_scaled,metric='euclidean', method ='average')))


# In[512]:


plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram_complete = shc.dendrogram((shc.linkage(df_scaled, metric='euclidean',method ='complete')))


# In[513]:


plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram_single = shc.dendrogram((shc.linkage(df_scaled,metric='euclidean', method ='single')))


# In[252]:


ac2 = AgglomerativeClustering(n_clusters = 4)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(df['Age'], df['Spending Score (1-100)'],
           c = ac2.fit_predict(df), cmap ='rainbow')
plt.show()


# In[ ]:




