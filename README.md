# 🧩 Customer Segmentation using Clustering

This project segments customers based on demographic and behavioral data to help businesses identify distinct user groups and improve marketing strategies.

## 🎯 Objective
To uncover spending patterns and derive actionable customer segments using unsupervised machine learning techniques.

## 📊 Dataset
- 2,240 real-world customer profiles  
- Features include demographic and behavioral attributes  
- Preprocessing: handled 24 missing values, removed 182 duplicates, scaled features, and encoded categorical variables

## ⚙️ Approach
- Applied **PCA (2 components)** for dimensionality reduction and visualization  
- Implemented **K-Means**, **DBSCAN**, and **Hierarchical Clustering** to segment customers  
- Evaluated models using **Silhouette Score (0.51)**  
- Derived **3 meaningful customer clusters** representing distinct spending behaviors

## 📈 Insights
- Clear segmentation based on income and spending patterns  
- Identified high-value and low-engagement customer groups  
- Visualized clusters using **Matplotlib** and **Seaborn**

## 🧠 Tech Stack
Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn

## 🚀 How to Run
```bash
pip install -r requirements.txt
python customer_segmentation.py
```

# 🧠 My Data Science App

This app visualizes data insights using Streamlit.

👉 **Try it here:** [Visit my Streamlit app](https://aakashchaudhari-customer-segmentation.streamlit.app)
