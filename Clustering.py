#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("transactions_n100000.csv")
df


# In[3]:


df.describe()


# In[4]:


# Number of unique ticket ids
len(df.ticket_id.unique())


# ## Choose Features (One Hot Encoder & Scaling)

# ### Transform Meal Type

# In[5]:


def meal_type(date):
    hr = int(date[-8:-6])
    if hr >= 20 or hr < 5:
        return "late_night_snack"
    elif hr >= 17:
        return "dinner"
    elif hr >= 14:
        return "afternoon_snack"
    elif hr >= 11:
        return "lunch"
    elif hr >= 5:
        return "breakfast"


# In[6]:


df["meal_type"] = df["order_timestamp"].apply(meal_type) 
df.head()


# In[7]:


# One Hot Encoder
ohe = OneHotEncoder(sparse=False)
transformed = ohe.fit_transform(df['meal_type'].to_numpy().reshape(-1, 1))

# Create a Pandas DataFrame of the hot encoded column
ohe_df_meal = pd.DataFrame(transformed, columns=ohe.get_feature_names())
ohe_df_meal = ohe_df_meal.rename(columns={"x0_afternoon_snack":"afternoon_snack","x0_dinner":"dinner",
                                    "x0_late_night_snack":"late_night_snack","x0_lunch": "lunch"})
                        

#concat with original data
df = pd.concat([df, ohe_df_meal], axis=1).drop("meal_type", axis=1)
df


# ### Transform Day of Week

# In[8]:


# Convert TimeStamp to day_of_week
df["order_timestamp"] = pd.to_datetime(df["order_timestamp"]) 
df['day_of_week'] = df['order_timestamp'].dt.day_name()


# In[9]:


# One Hot Encoder
ohe = OneHotEncoder(sparse=False)
transformed = ohe.fit_transform(df['day_of_week'].to_numpy().reshape(-1, 1))

# Create a Pandas DataFrame of the hot encoded column
ohe_df_day = pd.DataFrame(transformed, columns=ohe.get_feature_names())
ohe_df_day = ohe_df_day.rename(columns={"x0_Friday":"Fri","x0_Monday":"Mon","x0_Saturday":"Sat","x0_Sunday": "Sun",
                            "x0_Thursday": "Thu", "x0_Tuesday": "Tue", "x0_Wednesday": "Wed"})

#concat with original data
df = pd.concat([df, ohe_df_day], axis=1).drop("day_of_week", axis=1)
df


# ### Transform Item Name

# In[10]:


# One Hot Encoder
ohe = OneHotEncoder(sparse=False)
transformed = ohe.fit_transform(df['item_name'].to_numpy().reshape(-1, 1))

# Create a Pandas DataFrame of the hot encoded column
ohe_df = pd.DataFrame(transformed, columns=ohe.get_feature_names())
ohe_df = ohe_df.rename(columns={"x0_burger":"burger","x0_fries":"fries","x0_salad":"salad","x0_shake": "shake"})

#concat with original data
df = pd.concat([df, ohe_df], axis=1).drop(['item_name'], axis=1)
df


# ### Transform Location

# In[11]:


# One Hot Encoder
ohe = OneHotEncoder(sparse=False)
transformed = ohe.fit_transform(df['location'].to_numpy().reshape(-1, 1))

# Create a Pandas DataFrame of the hot encoded column
ohe_df_loc = pd.DataFrame(transformed, columns=ohe.get_feature_names())
ohe_df_loc = ohe_df_loc.rename(columns={"x0_1":"loc_1","x0_2":"loc_2","x0_3":"loc_3","x0_4": "loc_4","x0_5": "loc_5",
                                    "x0_6":"loc_6","x0_7":"loc_7","x0_8":"loc_8","x0_9": "loc_9"})

#concat with original data
df = pd.concat([df, ohe_df_loc], axis=1).drop(['location'], axis=1)
df


# In[12]:


pd.set_option('display.max_columns', None)
df


# In[13]:


df['sum_item_count'] = df.groupby("ticket_id")['item_count'].transform("sum")
df['avg_item_count'] = df.groupby("ticket_id")['item_count'].transform("mean")
df['burger'] = df.groupby("ticket_id")['burger'].transform("sum")
df['fries'] = df.groupby("ticket_id")['fries'].transform("sum")
df['salad'] = df.groupby("ticket_id")['salad'].transform("sum")
df['shake'] = df.groupby("ticket_id")['shake'].transform("sum")

df_uni = df.drop_duplicates("ticket_id").reset_index(drop=True)
df_uni


# ### Scaling

# In[14]:


# Scaling
from sklearn.preprocessing import StandardScaler
col_names = ['sum_item_count','avg_item_count']
features = df_uni[col_names]

scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

df_uni[col_names] = features
df_uni


# In[15]:


df_uni.shape


# ## Apply Clustering Algorithm

# In[16]:


df_clus = df_uni.drop(["order_timestamp","ticket_id","lat","long"], axis=1)
df_clus.head()


# In[17]:


df_clus.shape


# ### Elbow Test

# In[18]:


def get_wcss(df):
  wcss = []
  for i in range(1, 21):
      kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
      kmeans.fit(df)
      wcss.append(kmeans.inertia_)
  plt.plot(range(1, 21), wcss)
  plt.title('Elbow Method')
  plt.xlabel('Number of clusters')
  plt.ylabel('WCSS')
  plt.show()


# In[19]:


# determine k - look at wcss for each number of clustrs to determine how number of clusters
get_wcss(df_clus)


# ### Predict Clusters

# In[20]:


# create model
nclusters = 3

kmeans = KMeans(n_clusters=nclusters, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(df_clus)
cluster = kmeans.fit_predict(df_clus)


# In[21]:


def cluster_summary(df, cluster = cluster):

    cluster_summary = df.copy()

    cluster_summary['cluster'] = cluster

    # counts
    cluster_counts = cluster_summary['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['cluster', 'count']

    #means
    cluster_summary = cluster_summary.groupby('cluster').agg('mean').reset_index()
    cluster_summary = cluster_summary.merge(cluster_counts, on ='cluster')
    
    return cluster_summary


# In[22]:


pd.set_option('display.max_columns', None)
df_cluster_summary = cluster_summary(df_clus, cluster)
df_cluster_summary


# In[23]:


df_cluster_summary.to_csv('/Users/huangyurong/Desktop/Marketing Analytics/cluster_summary.csv', index = False, header=True)


# In[27]:


#dataframe with values

df_cluster = pd.DataFrame({'ticket_id': df_uni.index.values})

df_cluster['cluster'] = np.where(cluster==0, "Lunch_Salad_Loc5&8",
                                             np.where(cluster==1, "Dinner_More_Item_Shake&Burger_Loc4&7&9",
                                                      "Late Night_Burger_Loc2&6"))

df_cluster


# In[28]:


df_cluster.groupby("cluster").size()

