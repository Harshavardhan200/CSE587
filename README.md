# CSE587 Project: End-to-End Big Data Pipeline using Hadoop and Spark (Phase 1)

## Project Overview
This project is part of **CSE587: Big Data Processing**, focusing on designing and implementing an **end-to-end big data pipeline** using Hadoop and Spark.  
The dataset used is **eCommerce behavior data from a multi-category store**, sourced from [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).  

## Dataset Description
- The dataset contains user interactions on an eCommerce website, including views, adds to cart, and purchases.
- Data is spread across multiple CSV files, including:
  - **2019-Oct.csv** (~5GB)
  - **2019-Nov.csv** (~9GB)

## Project Steps

### 1. Data Download and Preprocessing
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).
- Extract the CSV files and place them in the local filesystem.

### 2. Exploratory Data Analysis (EDA)
Performed EDA using **Pandas** in a Jupyter Notebook:
- Checked missing values and handled inconsistencies.
- Generated summary statistics.
- Created visualizations for user interactions (e.g., views vs. purchases).
- Identified key trends in eCommerce behavior.

### 3. Hadoop Setup and Data Ingestion
- **Hadoop Cluster Setup**  
  Used a `docker-compose.yml` file to launch a **local Hadoop cluster** with:
  - NameNode
  - DataNode
  - ResourceManager
  - NodeManagers

- **Data Upload to HDFS**  
  Merged large files (`2019-Oct.csv`, `2019-Nov.csv`) and ingested them into HDFS using the following steps:
  ```bash
  hdfs dfs -mkdir -p /user/hadoop/ecommerce
  hdfs dfs -put 2019-Oct.csv /user/hadoop/ecommerce/
  hdfs dfs -put 2019-Nov.csv /user/hadoop/ecommerce/
  hdfs dfs -ls /user/hadoop/ecommerce/
