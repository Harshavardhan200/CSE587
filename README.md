# CSE587 Project: End-to-End Big Data Pipeline using Hadoop and Spark (Phase 1)

## Project Overview
This project is part of **CSE587: Big Data Processing**, focusing on designing and implementing an **end-to-end big data pipeline** using Hadoop and Spark.  
The dataset used is **eCommerce behavior data from a multi-category store**, sourced from [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).  

---

## Dataset Description
- The dataset contains user interactions on an eCommerce website, including views, adds to cart, and purchases.
- Data is spread across multiple CSV files, including:
  - **2019-Oct.csv** (~5GB)
  - **2019-Nov.csv** (~9GB)

---

## Project Steps

### 1. Download Dataset and Copy to Docker Hadoop

#### **Step 1: Download Dataset**
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).
- Extract CSV files and store them in the local filesystem.

#### **Step 2: Copy Dataset to Docker Hadoop**
Run the following command to copy data from the **local system to the Docker container running Hadoop**:

```bash
docker cp 2019-Oct.csv namenode:/input/
docker cp 2019-Nov.csv namenode:/input/

hdfs dfs -mkdir /input
hdfs dfs -put 2019-Oct.csv /input/
hdfs dfs -put 2019-Nov.csv /input/
hdfs dfs -ls /input/

hdfs dfs -get /input/merged_2019.csv /tmp/merged_2019.csv
