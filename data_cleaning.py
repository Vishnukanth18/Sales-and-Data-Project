#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


trxn=pd.read_csv('Retail_Data_Transactions.csv')


# In[5]:


trxn


# In[6]:


response=pd.read_csv('Retail_Data_Response.csv')


# In[7]:


response


# In[8]:


df=trxn.merge(response, on='customer_id', how='left')
df


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[11]:


df=df.dropna()


# In[12]:


df


# In[13]:


df.dtypes
df.shape
df.tail()


# In[14]:


df.describe()


# In[15]:


df.isnull().sum()


# In[16]:


df=df.dropna()


# In[17]:


df


# In[20]:


df['trans_date']= pd.to_datetime(df['trans_date'])
df['response']= df['response'].astype('int64')


# In[22]:


df


# In[23]:


set(df['response'])


# In[24]:


df.dtypes


# In[25]:


from scipy import stats
import numpy as np

z_scores= np.abs(stats.zscore(df['tran_amount']))


threshold= 3

outliers= z_scores>threshold


print(df[outliers])


# In[26]:


from scipy import stats
import numpy as np

z_scores= np.abs(stats.zscore(df['response']))

threshold= 3

outliers= z_scores>threshold


print(df[outliers])


# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['tran_amount'])
plt.show()


# In[30]:


df['month']= df['trans_date'].dt.month


# In[31]:


df


# In[32]:


monthly_Sales= df.groupby('month')['tran_amount'].sum()
monthly_Sales= monthly_Sales.sort_values(ascending=False).reset_index().head(3)
monthly_Sales


# In[33]:


customer_counts= df['customer_id'].value_counts().reset_index()
customer_counts.columns=['customer_id','count']


top_5_cus= customer_counts.sort_values(by='count', ascending=False).head(5)
top_5_cus


# In[34]:


sns.barplot(x='customer_id',y='count',data=top_5_cus)


# In[35]:


customer_sales= df.groupby('customer_id')['tran_amount'].sum().reset_index()
customer_sales



top_5_sal= customer_sales.sort_values(by='tran_amount', ascending=False).head(5)
top_5_sal


# In[36]:


sns.barplot(x='customer_id',y='tran_amount',data=top_5_sal)


# In[37]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df['month_year'] = df['trans_date'].dt.to_period('M')
monthly_sales = df.groupby('month_year')['tran_amount'].sum()

monthly_sales.index = monthly_sales.index.to_timestamp()

plt.figure(figsize=(12,6)) 
plt.plot(monthly_sales.index, monthly_sales.values) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) 
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  
plt.xlabel('Month-Year')
plt.ylabel('Sales')
plt.title('Monthly Sales')
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.show()


# In[38]:



recency = df.groupby('customer_id')['trans_date'].max()


frequency = df.groupby('customer_id')['trans_date'].count()


monetary = df.groupby('customer_id')['tran_amount'].sum()


rfm = pd.DataFrame({'recency': recency, 'frequency': frequency, 'monetary': monetary})


# In[39]:


def segment_customer(row):
    if row['recency'].year >= 2012 and row['frequency'] >= 15 and row['monetary'] > 1000:
        return 'P0'
    elif (2011 <= row['recency'].year < 2012) and (10 < row['frequency'] <= 15) and (500 < row['monetary'] <= 1000):
        return 'P1'
    else:
        return 'P2'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)


# In[40]:


rfm


# In[41]:


set(rfm['Segment'])


# In[42]:


churn_counts = df['response'].value_counts()

churn_counts.plot(kind='bar')


# In[43]:


top_5_customers = monetary.sort_values(ascending=False).head(5).index


top_customers_df = df[df['customer_id'].isin(top_5_customers)]


top_customers_sales = top_customers_df.groupby(['customer_id', 'month_year'])['tran_amount'].sum().unstack(level=0)
top_customers_sales.plot(kind='line')


# In[ ]:




