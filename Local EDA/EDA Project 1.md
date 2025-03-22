```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
```

# Load the dataset


```python
data = pd.read_csv("2019-Oct.csv")
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_time</th>
      <th>event_type</th>
      <th>product_id</th>
      <th>category_id</th>
      <th>category_code</th>
      <th>brand</th>
      <th>price</th>
      <th>user_id</th>
      <th>user_session</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-10-01 00:00:00 UTC</td>
      <td>view</td>
      <td>44600062</td>
      <td>2103807459595387724</td>
      <td>NaN</td>
      <td>shiseido</td>
      <td>35.79</td>
      <td>541312140</td>
      <td>72d76fde-8bb3-4e00-8c23-a032dfed738c</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-10-01 00:00:00 UTC</td>
      <td>view</td>
      <td>3900821</td>
      <td>2053013552326770905</td>
      <td>appliances.environment.water_heater</td>
      <td>aqua</td>
      <td>33.20</td>
      <td>554748717</td>
      <td>9333dfbd-b87a-4708-9857-6336556b0fcc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-10-01 00:00:01 UTC</td>
      <td>view</td>
      <td>17200506</td>
      <td>2053013559792632471</td>
      <td>furniture.living_room.sofa</td>
      <td>NaN</td>
      <td>543.10</td>
      <td>519107250</td>
      <td>566511c2-e2e3-422b-b695-cf8e6e792ca8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-10-01 00:00:01 UTC</td>
      <td>view</td>
      <td>1307067</td>
      <td>2053013558920217191</td>
      <td>computers.notebook</td>
      <td>lenovo</td>
      <td>251.74</td>
      <td>550050854</td>
      <td>7c90fc70-0e80-4590-96f3-13c02c18c713</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-10-01 00:00:04 UTC</td>
      <td>view</td>
      <td>1004237</td>
      <td>2053013555631882655</td>
      <td>electronics.smartphone</td>
      <td>apple</td>
      <td>1081.98</td>
      <td>535871217</td>
      <td>c6bd7419-2748-4c56-95b4-8cec9ff8b80d</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 42448764 entries, 0 to 42448763
    Data columns (total 9 columns):
     #   Column         Dtype  
    ---  ------         -----  
     0   event_time     object 
     1   event_type     object 
     2   product_id     int64  
     3   category_id    int64  
     4   category_code  object 
     5   brand          object 
     6   price          float64
     7   user_id        int64  
     8   user_session   object 
    dtypes: float64(1), int64(3), object(5)
    memory usage: 2.8+ GB
    


```python
data.columns
```




    Index(['event_time', 'event_type', 'product_id', 'category_id',
           'category_code', 'brand', 'price', 'user_id', 'user_session'],
          dtype='object')



## Dataset Overview

The dataset consists of **42,448,764** entries and **9** columns. Below is a detailed description of each column:

| # | Column Name    | Data Type | Description |
|---|--------------|-----------|-------------|
| 0 | **event_time** | object (string) | The timestamp when the event occurred. It should be converted to a datetime format for better analysis. |
| 1 | **event_type** | object (string) | Describes the type of user action, such as "view", "cart", or "purchase". |
| 2 | **product_id** | int64 | A unique identifier for each product. |
| 3 | **category_id** | int64 | A numerical ID representing the product category. |
| 4 | **category_code** | object (string) | A hierarchical category code that describes the product category in a readable format. Some values may be missing. |
| 5 | **brand** | object (string) | The brand name of the product. Some values may be missing. |
| 6 | **price** | float64 | The price of the product in the dataset’s currency. |
| 7 | **user_id** | int64 | A unique identifier for each user. This helps track user behavior across sessions. |
| 8 | **user_session** | object (string) | A unique identifier for a user’s session, used to group events performed by the same user in a single session. |

### Memory Usage
The dataset occupies approximately **2.8 GB** in memory, which is relatively large. Optimizing data types, such as converting `event_time` to datetime and using categorical data types for `event_type`, `category_code`, and `brand`, could help reduce memory consumption.



```python
data.shape
```




    (42448764, 9)



# Handle the null values


```python
data.isnull().sum()
```




    event_time              0
    event_type              0
    product_id              0
    category_id             0
    category_code    13515609
    brand             6117080
    price                   0
    user_id                 0
    user_session            2
    dtype: int64




```python
data = data.dropna()
```

# Date time format


```python
data['event_time'] = pd.to_datetime(data['event_time'])

data['year'] = data['event_time'].dt.year
data['month'] = data['event_time'].dt.month
data['day'] = data['event_time'].dt.day
data['dayofweek'] = data['event_time'].dt.dayofweek
```

## Data Cleaning: Column Removal

To optimize the dataset and improve analysis efficiency, we removed three columns: **`event_time`**, **`user_id`**, and **`user_session`**. Below is the explanation for why each of these columns was removed.

### 1. Removing `event_time`
- The `event_time` column stores the timestamp of each event.
- For many analyses, such as product popularity or user behavior, the exact timestamp may not be necessary.
- If time-based analysis is needed, we can extract only the **date** component from `event_time` instead of storing the full timestamp.
- Removing this column helps **reduce memory usage** and **simplify the dataset**.

### 2. Removing `user_id`
- The `user_id` column represents unique users in the dataset.
- While useful for tracking individual user behavior, it is **not essential** for aggregated analyses, such as product popularity or category trends.
- If user behavior analysis is needed, `user_session` can be used instead to study session-based interactions.
- Dropping this column helps **preserve user privacy** and **reduces data size**.

### 3. Removing `user_session`
- The `user_session` column groups multiple events performed by the same user within a single session.
- While it is useful for session-based analysis, it might not be necessary for all types of EDA (Exploratory Data Analysis).
- If needed, session-based trends can still be inferred from other columns, such as event type and product interactions.
- Removing this column **simplifies the dataset** while retaining essential insights.

### Final Outcome
After removing these three columns, the dataset now focuses on **product-related and category-level insights**, making it more efficient for further analysis. This step also helps in reducing memory usage, making the dataset easier to process in large-scale computations.



```python
data_cleaned = data.drop(columns=['event_time', 'user_id', 'user_session'])

```


```python
data_cleaned.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>product_id</th>
      <td>26560620.0</td>
      <td>5.366485e+06</td>
      <td>8.349450e+06</td>
      <td>1.000978e+06</td>
      <td>1.004856e+06</td>
      <td>1.600378e+06</td>
      <td>5.100337e+06</td>
      <td>6.050001e+07</td>
    </tr>
    <tr>
      <th>category_id</th>
      <td>26560620.0</td>
      <td>2.055725e+18</td>
      <td>1.572445e+16</td>
      <td>2.053014e+18</td>
      <td>2.053014e+18</td>
      <td>2.053014e+18</td>
      <td>2.053014e+18</td>
      <td>2.173217e+18</td>
    </tr>
    <tr>
      <th>price</th>
      <td>26560620.0</td>
      <td>3.528579e+02</td>
      <td>3.810814e+02</td>
      <td>8.800000e-01</td>
      <td>1.081100e+02</td>
      <td>2.185100e+02</td>
      <td>4.590900e+02</td>
      <td>2.574070e+03</td>
    </tr>
    <tr>
      <th>year</th>
      <td>26560620.0</td>
      <td>2.019000e+03</td>
      <td>0.000000e+00</td>
      <td>2.019000e+03</td>
      <td>2.019000e+03</td>
      <td>2.019000e+03</td>
      <td>2.019000e+03</td>
      <td>2.019000e+03</td>
    </tr>
    <tr>
      <th>month</th>
      <td>26560620.0</td>
      <td>1.000000e+01</td>
      <td>0.000000e+00</td>
      <td>1.000000e+01</td>
      <td>1.000000e+01</td>
      <td>1.000000e+01</td>
      <td>1.000000e+01</td>
      <td>1.000000e+01</td>
    </tr>
    <tr>
      <th>day</th>
      <td>26560620.0</td>
      <td>1.608226e+01</td>
      <td>8.658159e+00</td>
      <td>1.000000e+00</td>
      <td>9.000000e+00</td>
      <td>1.600000e+01</td>
      <td>2.300000e+01</td>
      <td>3.100000e+01</td>
    </tr>
    <tr>
      <th>dayofweek</th>
      <td>26560620.0</td>
      <td>2.967237e+00</td>
      <td>1.957543e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>3.000000e+00</td>
      <td>5.000000e+00</td>
      <td>6.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_cleaned.event_type.unique()
```




    array(['view', 'purchase', 'cart'], dtype=object)



# Event TYPE pie chart with percentage of event type for each


```python
labels = ['view', 'cart','purchase']
size = data_cleaned['event_type'].value_counts()
colors = ['green', 'blue','red']
explode = [0, 0.1,0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Event_Type', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
```


    
![png](output_18_0.png)
    


# Top 10 brands by event type


```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
event_types = ['view', 'cart', 'purchase']
colors = ['magma', 'viridis', 'coolwarm']

top_brands_by_event = {event: data_cleaned[data_cleaned['event_type'] == event]['brand'].value_counts().head(10) for event in event_types}

for i, event in enumerate(event_types):
    sns.barplot(y=top_brands_by_event[event].index, x=top_brands_by_event[event].values, palette=colors[i], ax=axes[i])
    axes[i].set_title(f"Top 10 Brands - {event.capitalize()}")
    axes[i].set_xlabel("Number of Events")
    axes[i].set_ylabel("Brand")

plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    


# Top 10 popular categories


```python
category_counts = data_cleaned['category_code'].value_counts().head(10)
plt.figure(figsize=(12, 6))
squarify.plot(sizes=category_counts.values, label=category_counts.index, alpha=0.7, color=sns.color_palette("coolwarm", 10))
plt.title("Top 10 Popular Categories")
plt.axis('off')
plt.show()

```


    
![png](output_22_0.png)
    


# Label Encoding 


```python
mapping = {'view': 0, 'cart': 1, 'purchase': 2}
data_cleaned['event_type'] = data_cleaned['event_type'].map(mapping)
data_cleaned
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_type</th>
      <th>product_id</th>
      <th>category_id</th>
      <th>category_code</th>
      <th>brand</th>
      <th>price</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3900821</td>
      <td>2053013552326770905</td>
      <td>appliances.environment.water_heater</td>
      <td>aqua</td>
      <td>33.20</td>
      <td>2019</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1307067</td>
      <td>2053013558920217191</td>
      <td>computers.notebook</td>
      <td>lenovo</td>
      <td>251.74</td>
      <td>2019</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1004237</td>
      <td>2053013555631882655</td>
      <td>electronics.smartphone</td>
      <td>apple</td>
      <td>1081.98</td>
      <td>2019</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1480613</td>
      <td>2053013561092866779</td>
      <td>computers.desktop</td>
      <td>pulser</td>
      <td>908.62</td>
      <td>2019</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>28719074</td>
      <td>2053013565480109009</td>
      <td>apparel.shoes.keds</td>
      <td>baden</td>
      <td>102.71</td>
      <td>2019</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>42448757</th>
      <td>0</td>
      <td>1004870</td>
      <td>2053013555631882655</td>
      <td>electronics.smartphone</td>
      <td>samsung</td>
      <td>275.25</td>
      <td>2019</td>
      <td>10</td>
      <td>31</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42448758</th>
      <td>0</td>
      <td>2702331</td>
      <td>2053013563911439225</td>
      <td>appliances.kitchen.refrigerators</td>
      <td>lg</td>
      <td>527.43</td>
      <td>2019</td>
      <td>10</td>
      <td>31</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42448759</th>
      <td>0</td>
      <td>2300275</td>
      <td>2053013560530830019</td>
      <td>electronics.camera.video</td>
      <td>gopro</td>
      <td>527.40</td>
      <td>2019</td>
      <td>10</td>
      <td>31</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42448761</th>
      <td>0</td>
      <td>5701038</td>
      <td>2053013553970938175</td>
      <td>auto.accessories.player</td>
      <td>kenwood</td>
      <td>128.70</td>
      <td>2019</td>
      <td>10</td>
      <td>31</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42448762</th>
      <td>0</td>
      <td>21407424</td>
      <td>2053013561579406073</td>
      <td>electronics.clocks</td>
      <td>tissot</td>
      <td>689.85</td>
      <td>2019</td>
      <td>10</td>
      <td>31</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>26560620 rows × 10 columns</p>
</div>



# Countplotting based on the event_type


```python
sns.countplot(x=data_cleaned['event_type'])
plt.title("Distribution of Event Type")
plt.show()
```


    
![png](output_26_0.png)
    


# Prepare the box plot


```python
plt.figure(figsize=(10,5))
sns.boxplot(x='event_type', y='price', data=data_cleaned)
plt.title("Price Distribution by Event Type")
plt.show()

```


    
![png](output_28_0.png)
    


# Remove the outliers


```python
Q1 = data_cleaned['price'].quantile(0.25)
Q3 = data_cleaned['price'].quantile(0.75)

# Compute IQR
IQR = Q3 - Q1

# Define the bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
data_cleaned = data_cleaned[(data_cleaned['price'] >= lower_bound) & (data_cleaned['price'] <= upper_bound)]


plt.figure(figsize=(10,5))
sns.boxplot(x='event_type', y='price', data=data_cleaned)
plt.title("Price Distribution by Event Type")
plt.show()

```


    
![png](output_30_0.png)
    



```python
plt.figure(figsize=(10, 5))
sns.countplot(data=data_cleaned, x='dayofweek', palette='husl')
plt.title("User Activity by Day of the Week")
plt.xlabel("Day of the Week (0 = Monday, 6 = Sunday)")
plt.ylabel("Event Count")
plt.show()
```


    
![png](output_31_0.png)
    



```python

```
