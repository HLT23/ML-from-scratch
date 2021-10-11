#!/usr/bin/env python
# coding: utf-8

# In[234]:


import pandas as pd
import matplotlib.pyplot as plt;
import numpy as np;
import collections as col;
from sklearn.decomposition import PCA;


# In[235]:


headers = ["gene_id", "ground_truth","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]
cho_df = pd.read_table('/Users/haydentrautmann/Desktop/Python projects/347projects/cho.txt',names=headers)


# In[236]:


cho_df


# In[237]:


headers = ["gene_id", "ground_truth","A","B","C","D","E","F","G","H","I","J","K","L"]
iyer_df = pd.read_table('/Users/haydentrautmann/Desktop/Python projects/347projects/iyer.txt', names=headers)


# In[238]:


iyer_df


# In[239]:


cho_dups = cho_df.duplicated()
iyer_dups = iyer_df.duplicated()
print('number of cho duplicate rows = %d' % cho_dups.sum())
print('number of iyer duplicate rows = %d' % iyer_dups.sum())


# In[240]:


#remove outliers identified as -1 values in ground truth clusters
cho_df = cho_df[cho_df.ground_truth != -1]
iyer_df = iyer_df[iyer_df.ground_truth != -1]


# In[241]:


#remove gene_id
cho_df = cho_df.drop(columns="gene_id")
iyer_df = iyer_df.drop(columns="gene_id")


# In[242]:


#plt.scatter(cho.column,cho.row)


# In[243]:


cho = cho_df.to_numpy();
iyer = iyer_df.to_numpy();
print("got here");


# In[244]:


#PCA reduction into 2 components, also scale the data
pca=PCA(n_components=2).fit(cho);
cho=pca.transform(cho);

pca=PCA(n_components=2).fit(iyer);
iyer=pca.transform(iyer);


# In[ ]:





# In[381]:


def kmeans(dataset):

    #randomly select k=3 centroids for initialization
    #centroids are random elements in the array
    index1 = np.random.randint(dataset.shape[0]);
    index2 = np.random.randint(dataset.shape[0]);
    index3 = np.random.randint(dataset.shape[0]);

    centroid1 = dataset[index1];
    centroid2 = dataset[index2];
    centroid3 = dataset[index3];

    #initializing dictionary which will store gene ids in cluster with closest centroid

    clust1 = [];
    clust2 = [];
    clust3 = [];

    #temp_clust["clust_c1"] = [];
    #temp_clust["clust_c2"] = [];
    #temp_clust["clust_c3"] = [];

    #it = np.nditer(cho);
    #for x in it:

    # Find Euclidean Distances from each row with all its attributes to the random clusters
    # Classify cluster for row based on closest cluster
    # Store rows in dict() with keys as assigned cluster, and values as original row
    for i,x in enumerate(dataset):
        dist_c1 = np.linalg.norm(x - centroid1)
        dist_c2 = np.linalg.norm(x - centroid2)
        dist_c3 = np.linalg.norm(x - centroid3)

        minimum = min(dist_c1, dist_c2, dist_c3);
        if (minimum == dist_c1):
            clust1.append(x);
        if (minimum == dist_c2):
            clust2.append(x);
        else:
            clust3.append(x);

    # Recompute each centroid 
    # Gets mean of all gene id's collumns in given cluster
    centroid1 = np.mean(clust1,axis=0);
    centroid2 = np.mean(clust2,axis=0);
    centroid3 = np.mean(clust3,axis=0);

    # Repeat previous Euclidean Distance process to classify Gene IDs with recomputed centroids
    # Recalculate centroids 100 times
    j = 1
    while j <= 100:
        for i,x in enumerate(dataset):
            dist_c1 = np.linalg.norm(x - centroid1)
            dist_c2 = np.linalg.norm(x - centroid2)
            dist_c3 = np.linalg.norm(x - centroid3)

            minimum = min(dist_c1, dist_c2, dist_c3);
            if (minimum == dist_c1):
                clust1.append(x);
            if (minimum == dist_c2):
                clust2.append(x);
            else:
                clust3.append(x);

        # Recompute each centroid 
        # Gets mean of all gene id's collumns in given cluster
        centroid1 = np.mean(clust1,axis=0);
        centroid2 = np.mean(clust2,axis=0);
        centroid3 = np.mean(clust3,axis=0);
        j += 1;
    clusters = []
    clusters.append(clust1)
    clusters.append(clust2)
    clusters.append(clust3)
    return clusters


# In[382]:


# Print Gene IDs in given cluster after 10 iterations

def graph(clusters):

    
    
    i=0;


    colx1 = tuple(x[0] for x in clusters[2])
    coly2 = tuple(x[1] for x in clusters[2])
    plt.scatter(colx1, coly2, color= "black")
    #plt.show()
    colx3 = tuple(x[0] for x in clusters[0])
    coly4 = tuple(x[1] for x in clusters[0])
    plt.scatter(colx3, coly4, color= "red")

    colx5 = tuple(x[0] for x in clusters[1])
    coly6 = tuple(x[1] for x in clusters[1])
    plt.scatter(colx5, coly6, color= "green")


# In[247]:


cluster_cho = kmeans(cho)
graph(cluster_cho)


# In[248]:


cluster_iyer = kmeans(iyer)
graph(cluster_iyer)


# In[ ]:





# In[383]:


#Spectral Clustering Implementation
# Step 1: Representing data points as symmetric similarity graph
# Chose k-NN for this similarity graph
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def spectral_step1(dataset):
    #isolate X and y values from dataset
    #let y be the ground truth values
#     print(dataset.shape[0])
#     i=0;
    scale = StandardScaler()  
    X = np.array(dataset.iloc[:,1:-1])
    y = np.array(dataset['ground_truth'])
#     length = len(dataset)
#     i=0;
#     x = []
#     y = []
#     while i < length:
#         x.append(dataset[i][0])
#         y.append(dataset[i][1]) 
#         i+=1;

    global x_train,x_test, y_train, y_test; 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    knn = KNeighborsClassifier(n_neighbors = 5)
   
    #scaling train and test data
    scale.fit(x_train)
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)
    
    knn.fit(x_train,y_train)
    y_predict = knn.predict(x_test)
    print(classification_report(y_test, y_predict))
    matrix = confusion_matrix(y_test, y_predict)
    print(matrix)
    return matrix
#     #knn = knn.fit()
#     predict = knn.predict(x_test)
#     print(classification_report(y_test_data, predict))
#     print(confusion_matrix(y_test_data, predict))
    


# In[384]:


iyer_simil = spectral_step1(iyer_df)


# In[430]:


cho_simil = spectral_step1(cho_df)


# In[431]:


from scipy.sparse import csgraph

#Steps 2 and 3 for spectral clustering
# computes graph laplacian then 
def spectral_step23(similarity):
    lap = csgraph.laplacian(similarity)
    
    #compute k eigenvectors corresponding to the k-smallest non-zero eigenvalues of L
    e_val, e_vec = np.linalg.eig(lap)
    #     e_val, e_vec = np.linalg.eig(np.matrix(lap))

#     eig = egval.real.argsort()[:3]
#     #vect = np.ndarray(shape=(lap.shape[0],0))
#     for i in range(1, eig.shape[0]):
#         new_matrix = np.transpose(np.matrix(egvec[:,eig[i].item()]))
#         #vect = np.(vect, new_matrix),axis=0))
#     return new_matrix
        
#     np.where(e_val == np.partition(e_val, 1)[1])
#     eig = e_vec[:,1].copy()
#     eig[eig < 0] = 0
#     eig[eig > 0] = 1
#     type(eig),y_test.shape,eig.shape
#     eig:
#         print(eig[i])
    print(lap)
    print(e_val)
    print(e_vec)
    print(type(e_vec))
    return e_vec
    


# In[432]:


eigenvec = spectral_step23(cho_simil)


# In[433]:


def pca(array):
    pca=PCA(n_components=2).fit(array);
    new_arr=pca.transform(array);
    return new_arr


# In[434]:


# Spectral Clustering Step 4
# Cluster the eigenvectors into K-Means with K clusters
eigen_2d = pca(eigenvec)
spectral = kmeans(eigen_2d)


# In[435]:


graph(spectral)


# In[429]:


simil_iyer = spectral_step1(iyer_df)
iyer_eigen = spectral_step23(simil_iyer)
eigen_2d = pca(iyer_eigen)
iyer_spec = kmeans(eigen_2d)
graph(iyer_spec)


# In[ ]:




