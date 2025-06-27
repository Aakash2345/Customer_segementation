import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import pickle
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("data//csv files//pca.csv") # Preprocessed data till pca

# Original scale data after applying k-means 
summary_data = pd.read_csv("data//csv files//inverse_transformed_data.csv") # data after inverse tranformation of log and scaling

# Load the preprocess model (OneHotEncoding -> Scaling -> PCA)
model_path1 = r'data\models\preprocess_model.pkl'
with open(model_path1, 'rb') as file1:
    preprocess_model = pickle.load(file1)
    
# use dbscan to remove outliers 
model = DBSCAN(eps = 0.5, min_samples = 5)
db_labels = model.fit_predict(data)

# Exclude noise points labeled as -1
mask = db_labels != -1

# Load the K-Means model 
model_path2 = r'data\models\K_means_model.pkl'
with open(model_path2, 'rb') as file2:
    Cluster_model = pickle.load(file2)
 
# Predicting the pca data   
pre_label = Cluster_model.predict(data)
data['K_Cluster'] = pre_label

st.title("Customer Cluster Prediction")

# Different tabs for summary, prediction and visualization
tab0, tab1, tab2 = st.tabs(["Summary","Predict Cluster", "Visualize"])

# Tab0
with tab0:
    st.subheader("Cluster Summary")
    cluster_ids = sorted(summary_data['K_cluster'].unique())

    for cluster_id in cluster_ids:
        cluster_data = summary_data[(summary_data['K_cluster'] == cluster_id) & mask]
        if not cluster_data.empty:
            avg_spend = cluster_data['TotalMntprod'].mean()
            avg_income = cluster_data['Income'].mean()
            avg_age = cluster_data['Age'].mean()
            avg_recency = cluster_data['Recency'].mean()
            num_customers = len(cluster_data)
            
            st.markdown(f"### Cluster {cluster_id}")
            col1, col2, col3, col4, col5 = st.columns(5) # Specifying 5 columns 

            col1.metric("Customers", f"{num_customers}")
            col2.metric("Avg Income", f"₹{avg_income:.0f}")
            col3.metric("Avg Age", f"{avg_age:.1f}")
            col4.metric("Avg Spend", f"₹{avg_spend:.0f}")
            col5.metric("Avg Recency", f"{avg_recency:.0f} days")

            st.markdown("---")  # horizontal line for separation

# Tab1
with tab1:
    # Works like dropdown
    with st.expander("Spending on Products"):
        mnt_wines = st.number_input("Amount spent on Wines", min_value=0)
        mnt_fruits = st.number_input("Amount spent on Fruits", min_value=0)
        mnt_meat = st.number_input("Amount spent on Meat Products", min_value=0)
        mnt_fish = st.number_input("Amount spent on Fish Products", min_value=0)
        mnt_sweets = st.number_input("Amount spent on Sweet Products", min_value=0)
        mnt_gold = st.number_input("Amount spent on Gold Products", min_value=0)
    
    with st.expander("Purchase & Visit Behavior"):
        num_deals_purchases = st.number_input("Number of Deals Purchases", min_value=0, step = 1)
        num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, step = 1)
        num_catalog_purchases = st.number_input("Number of Catalog Purchases", min_value=0, step = 1)
        num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, step = 1)
        num_web_visits_month = st.number_input("Number of Web Visits per Month", min_value=0, step = 1)

    # Select the education level
    education = st.selectbox(
        "Education Level",
        options=['Graduation', 'PhD', 'Master', 'Basic', "2n Cycle (Bachelor's degree or equivalent)"]
    )

    # Select the marital_status
    marital_status = st.selectbox(
        "Marital Status",
        options=['Single', 'Together', 'Married', 'Divorced', 'Widow']
    )

    # Select the multiple campaigns
    with st.expander("Campaigns Accepted"):
        accepted_campaigns = st.multiselect(
        "Select campaigns accepted by customer:",
        options=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    )

    # Then convert to 0/1 features for each campaign
    accepted_cmp1 = 1 if 'AcceptedCmp1' in accepted_campaigns else 0
    accepted_cmp2 = 1 if 'AcceptedCmp2' in accepted_campaigns else 0
    accepted_cmp3 = 1 if 'AcceptedCmp3' in accepted_campaigns else 0
    accepted_cmp4 = 1 if 'AcceptedCmp4' in accepted_campaigns else 0
    accepted_cmp5 = 1 if 'AcceptedCmp5' in accepted_campaigns else 0

    with st.expander("Personal Information"):
        income = st.number_input("Income", min_value = 1000) 
        if marital_status == 'Single':
            st.info("Since the person is single, no kids or teens at home assumed.")
            kid_home = st.number_input("Number of Kids at Home", min_value=0, step=1, value=0, disabled=True)
            teen_home = st.number_input("Number of Teens at Home", min_value=0, step=1, value=0, disabled=True)
        else:
            kid_home = st.number_input("Number of Kids at Home", min_value=0, step=1)
            teen_home = st.number_input("Number of Teens at Home", min_value=0, step=1)
        recency = st.number_input("Recency (days since last purchase)", min_value=0, step = 1)
        complain = st.selectbox("Complaint Registered?", options=[0, 1])
        age = st.number_input("Age", min_value = 18, max_value = 100, step = 1)
    
    # Mapping user input to variables 
    input_dict = {
        'Education': education, 'Marital_Status': marital_status, 'Income': income, 'Kidhome': kid_home,
        'Teenhome': teen_home, 'Recency': recency, 'MntWines': mnt_wines, 'MntFruits': mnt_fruits,
        'MntMeatProducts': mnt_meat, 'MntFishProducts': mnt_fish,'MntSweetProducts': mnt_sweets,
        'MntGoldProds': mnt_gold, 'NumDealsPurchases': num_deals_purchases, 'NumWebPurchases': num_web_purchases,
        'NumCatalogPurchases': num_catalog_purchases, 'NumStorePurchases': num_store_purchases,
        'NumWebVisitsMonth': num_web_visits_month, 'AcceptedCmp3': accepted_cmp3, 'AcceptedCmp4': accepted_cmp4,
        'AcceptedCmp5': accepted_cmp5, 'AcceptedCmp1': accepted_cmp1, 'AcceptedCmp2': accepted_cmp2,
        'Complain': complain, 'Age': age
    }

    # Converting to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Function to predict cluster
    def predict_cluster(input_df, preprocess_model, Cluster_model):
        processed_input = preprocess_model.transform(input_df) # Preprocess the data
        cluster = Cluster_model.predict(processed_input) # Apply K-Means to preprocess data
        return cluster[0]

    # Creating prediction button
    if st.button("Predict Cluster"):
        cluster_result = predict_cluster(input_df, preprocess_model, Cluster_model)
        st.success(f'Predicted Customer Cluster : {cluster_result}') # Displays the predicted cluster

# Tab2        
with tab2:
    new_point_pca = preprocess_model.transform(input_df) # new data point for visualization
    fig ,ax = plt.subplots() 
    if 'cluster_result' in locals(): # check if cluster_result exists or not
        # Get unique K-Means cluster IDs from non-outlier (inlier) points
        cluster_ids = sorted(data.loc[mask, 'K_Cluster'].unique())
        
        # Plot each cluster separately with its label
        for cluster_id in cluster_ids:
            # Select data points that are inliers (non-outliers) and belong to the current K-Means cluster
            cluster_points = data.loc[(mask) & (data['K_Cluster'] == cluster_id)]
            # Plot the PCA-reduced data points for the current K-Means cluster
            ax.scatter(
                cluster_points.iloc[:, 0], cluster_points.iloc[:, 1],
                label=f'Cluster {cluster_id}'
            )
        # Plotting new data point in same plot
        ax.scatter(
            new_point_pca[0, 0], new_point_pca[0, 1], 
            color = 'red', marker = 'o', s = 200, 
            label = f'New customer (cluster {cluster_result})'
        )
        ax.set_title("K-Means Clusters with New Point")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        
        # Define the cluster descriptions
        cluster_descriptions = {
            0: {
                "name": "Balanced Customers",
                "nature": "Moderate income, Moderate spending or both",
                "stay": "They appreciate consistent value, trust the brand, and benefit from seasonal offers and loyalty programs.",
                "keep": "Maintain their trust with consistent quality, timely promotions, and loyalty rewards that acknowledge their repeat engagement."
            },
            1: {
                "name": "Budget-Oriented Customers",
                "nature": "Lower income, Lower spenders or both",
                "stay": "They stay for affordability, essential product availability and discount-driven value.",
                "keep": "Offer value packs, flexible pricing, and frequent discounts. Ensure basic products are always accessible and affordable."
            },
            2: {
                "name": "Premium Customers",
                "nature": "High income, high spenders or both",
                "stay": "They stay for high-quality service and premium experiences.",
                "keep": "Provide personalized service, exclusive access to high-end products, and seamless premium experiences."
            }
        }

        # Fetch the description
        if cluster_result in cluster_descriptions:
            description = cluster_descriptions[cluster_result]
            st.subheader(f"Customer belongs to Cluster {cluster_result}: {description['name']}")
            st.write(f"**Nature:** {description['nature']}")
            st.write(f"**Why They Stay:** {description['stay']}")
            st.write(f"**How to Keep Them:** {description['keep']}")
        else:
            st.error("Invalid cluster result.")
    else:
        st.warning("Predict a cluster first using the Predict Cluster in Tab 1.")
