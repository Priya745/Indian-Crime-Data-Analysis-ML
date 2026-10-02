import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Indian Crime Data Analysis", layout="wide")

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_models():
    models_dir = "models"
    try:
        lr_model = joblib.load(os.path.join(models_dir, "linear_regression_model.joblib"))
        dt_model = joblib.load(os.path.join(models_dir, "decision_tree_model.joblib"))
        rf_model = joblib.load(os.path.join(models_dir, "random_forest_model.joblib"))
        knn_model = joblib.load(os.path.join(models_dir, "knn_model.joblib"))
        knn_scaler = joblib.load(os.path.join(models_dir, "knn_scaler.joblib"))
        kmeans_model = joblib.load(os.path.join(models_dir, "kmeans_model.joblib"))
        kmeans_scaler = joblib.load(os.path.join(models_dir, "kmeans_scaler.joblib"))
        metadata = joblib.load(os.path.join(models_dir, "model_metadata.joblib"))
        return lr_model, dt_model, rf_model, knn_model, knn_scaler, kmeans_model, kmeans_scaler, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}\nPlease ensure you have run the model saving step.")
        return None, None, None, None, None, None, None, None

@st.cache_data
def load_data():
    file_path = "data/factors/crime/01_District_wise_crimes_committed_IPC_2001_2012.csv"
    try:
        df = pd.read_csv(file_path)
        df_clean = df[~df['DISTRICT'].isin(['TOTAL', 'DELHI UT TOTAL'])].copy()
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {e}\nPlease ensure the CSV file is located at the correct path.")
        return None

lr_model, dt_model, rf_model, knn_model, knn_scaler, kmeans_model, kmeans_scaler, metadata = load_models()
df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Crime Prediction", "District Clustering", "Model Performance", "About Dataset"])

# --- PAGE 1: HOME ---
if page == "Home":
    st.title("Indian Crime Data Analysis & Machine Learning")
    st.write("An analytical application exploring historical reported crime patterns across Indian districts using regression, classification, and unsupervised clustering.")
    
    st.header("Project Overview")
    st.write("**Project Objective:** Analyze Indian districts based on historical reported crime patterns using regression, classification, and clustering to identify comparatively lower/higher reported crime levels and groups of districts with similar crime profiles.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical Period", "2001–2012")
    col2.metric("Cleaned Observations", "8,597")
    col3.metric("District Profiles (Clustering)", "~806")
    col4.metric("ML Models", "5")
    
    st.header("Available Models")
    st.markdown("""
    * **Linear Regression** → predicts continuous Violent Crime Score
    * **Decision Tree** → classifies Violent Crime Level
    * **Random Forest** → classifies Violent Crime Level
    * **KNN** → classifies Violent Crime Level
    * **K-Means** → groups districts with similar historical crime profiles
    """)
    st.info("Note: This app is an analytical/educational tool based purely on historical reported crime data. It does not classify any district as objectively 'safe', 'unsafe', 'dangerous', or 'safest'.")

# --- PAGE 2: CRIME PREDICTION ---
elif page == "Crime Prediction":
    st.title("Crime Prediction")
    st.write("Enter crime values manually to predict comparative reported crime levels based on historical models.")
    
    if metadata is None:
        st.stop()
        
    model_choice = st.selectbox("Select Model", ["Linear Regression", "Decision Tree", "Random Forest", "KNN"])
    
    st.markdown("### Enter Crime Values")
    
    if model_choice == "Linear Regression":
        st.write("This model uses 3 features to predict the continuous Violent Crime Score.")
        pc_score = st.number_input("Property Crime Score", min_value=0, value=100)
        cheating = st.number_input("CHEATING", min_value=0, value=20)
        riots = st.number_input("RIOTS", min_value=0, value=10)
        
        if st.button("Predict"):
            input_df = pd.DataFrame([[pc_score, cheating, riots]], columns=metadata['linear_regression_features'])
            try:
                pred = lr_model.predict(input_df)[0]
                st.success(f"### Predicted Violent Crime Score: {pred:.2f}")
                st.write("*The prediction represents the model's estimated Violent Crime Score based on the three input features. It is not a crime rate.*")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                
    else:
        st.write("This model uses 5 features to classify the Violent Crime Level.")
        col1, col2, col3 = st.columns(3)
        with col1:
            theft = st.number_input("THEFT", min_value=0, value=200)
            burglary = st.number_input("BURGLARY", min_value=0, value=50)
        with col2:
            robbery = st.number_input("ROBBERY", min_value=0, value=10)
            cheating = st.number_input("CHEATING", min_value=0, value=20)
        with col3:
            riots = st.number_input("RIOTS", min_value=0, value=10)
            
        if st.button("Predict"):
            input_df = pd.DataFrame([[theft, burglary, robbery, cheating, riots]], columns=metadata['classification_features'])
            
            try:
                if model_choice == "Decision Tree":
                    pred = dt_model.predict(input_df)[0]
                    proba = dt_model.predict_proba(input_df)[0] if hasattr(dt_model, "predict_proba") else None
                elif model_choice == "Random Forest":
                    pred = rf_model.predict(input_df)[0]
                    proba = rf_model.predict_proba(input_df)[0] if hasattr(rf_model, "predict_proba") else None
                elif model_choice == "KNN":
                    # Important: Must scale KNN input!
                    scaled_input = knn_scaler.transform(input_df)
                    pred = knn_model.predict(scaled_input)[0]
                    proba = knn_model.predict_proba(scaled_input)[0] if hasattr(knn_model, "predict_proba") else None
                
                st.success(f"### Predicted Violent Crime Level: {pred}")
                
                st.write("**Classification Thresholds Used (from 2001-2009 Training Data):**")
                st.code("Low: score <= 79.0\nMedium: 79.0 < score <= 168.0\nHigh: score > 168.0")
                
                if proba is not None:
                    st.write("**Model Class Probabilities:**")
                    classes = metadata.get('classification_classes', ['High', 'Low', 'Medium'])
                    # Ensure matching order for proba mapping
                    actual_classes = dt_model.classes_ if model_choice == "Decision Tree" else (rf_model.classes_ if model_choice == "Random Forest" else knn_model.classes_)
                    
                    prob_df = pd.DataFrame([proba], columns=actual_classes).round(4) * 100
                    prob_df = prob_df.astype(str) + "%"
                    st.dataframe(prob_df)
                    st.info("These are the model's mathematical estimated class probabilities based on training data, NOT real-world probabilities of crime occurring.")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")

# --- PAGE 3: DISTRICT CLUSTERING ---
elif page == "District Clustering":
    st.title("District Clustering")
    st.write("K-Means groups districts based on similarity in their historical average reported crime profiles.")
    
    if df is None or kmeans_model is None:
        st.stop()
        
    # Aggregate data to historical averages per district
    km_features = metadata['kmeans_features']
    district_profiles = df.groupby('DISTRICT')[km_features].mean().reset_index()
    
    districts_list = sorted(district_profiles['DISTRICT'].unique())
    selected_district = st.selectbox("Select a district", districts_list)
    
    if selected_district:
        st.subheader(f"Historical Average Profile: {selected_district}")
        dist_data = district_profiles[district_profiles['DISTRICT'] == selected_district]
        st.dataframe(dist_data[km_features].round(2))
        
        # Scale and predict
        scaled_dist = kmeans_scaler.transform(dist_data[km_features])
        cluster_assignment = kmeans_model.predict(scaled_dist)[0]
        
        st.success(f"### Cluster: {cluster_assignment}")
        st.write(f"*This district belongs to Cluster {cluster_assignment} based on similarity across the nine historical crime features.*")
        
        # Calculate cluster profiles (dynamically across all districts)
        scaled_all = kmeans_scaler.transform(district_profiles[km_features])
        district_profiles['Cluster'] = kmeans_model.predict(scaled_all)
        
        st.markdown("---")
        st.subheader(f"Profile of Cluster {cluster_assignment}")
        
        cluster_stats = district_profiles.groupby('Cluster')[km_features].mean().round(2).reset_index()
        cluster_sizes = district_profiles['Cluster'].value_counts().sort_index().rename("Number of districts")
        
        cluster_stats = cluster_stats.merge(cluster_sizes, left_on='Cluster', right_index=True)
        
        # Reorder columns to put Number of districts first
        cols = ['Cluster', 'Number of districts'] + km_features
        cluster_stats = cluster_stats[cols]
        
        st.write("Average historical reported crimes for all clusters:")
        st.dataframe(cluster_stats)
        
        # Bar chart for the selected cluster
        selected_cluster_data = cluster_stats[cluster_stats['Cluster'] == cluster_assignment][km_features].iloc[0]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(km_features, selected_cluster_data, color='teal')
        ax.set_title(f"Average Crime Counts for Cluster {cluster_assignment}")
        ax.set_ylabel("Average Count")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

# --- PAGE 4: MODEL PERFORMANCE ---
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("These metrics were obtained from the historical test sets used during model development (Testing Period: 2010–2012).")
    
    st.subheader("Classification Models (Target: Violent Crime Level)")
    clf_metrics = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest", "KNN"],
        "Accuracy": ["69.11%", "70.30%", "67.93%"],
        "Weighted Precision": ["69.56%", "69.84%", "67.42%"],
        "Weighted Recall": ["69.11%", "70.30%", "67.93%"],
        "Weighted F1": ["69.31%", "70.01%", "67.55%"]
    })
    st.table(clf_metrics)
    
    st.subheader("Regression Model (Target: Violent Crime Score)")
    reg_metrics = pd.DataFrame({
        "Model": ["Linear Regression"],
        "MAE": ["86.81"],
        "MSE": ["18,627.04"],
        "RMSE": ["136.48"],
        "R²": ["0.3719"]
    })
    st.table(reg_metrics)
    
    st.header("Model Interpretation")
    st.write("On the historical test set, the Random Forest, Decision Tree, and KNN models produced different classification performance. Their results should be interpreted strictly in the context of this dataset, feature selection, and the chronological train/test split. Random Forest provided a slight performance boost over the Decision Tree, while KNN struggled more with the class boundaries.")
    st.write("For Linear Regression, an R² of 0.3719 means that approximately 37.2% of the variation in the test-set Violent Crime Score was explained by the selected property/disturbance features under this historical split. This confirms that crime is not fully mathematically predictable from just a few features, especially without socioeconomic or demographic data.")

# --- PAGE 5: ABOUT DATASET ---
elif page == "About Dataset":
    st.title("About the Dataset")
    
    st.subheader("Dataset Details")
    st.write("**Dataset:** District-wise reported IPC crimes.")
    st.write("**Source file:** `01_District_wise_crimes_committed_IPC_2001_2012.csv`")
    st.write("**Period:** 2001–2012.")
    st.write("**Observation Level:** Each original row represents reported crime counts for a specific district/state-year combination.")
    
    st.subheader("Data Cleaning")
    st.write("Summary rows containing aggregate totals were removed to prevent data leakage and clustering distortion:")
    st.code('- DISTRICT == "TOTAL"\n- DISTRICT == "DELHI UT TOTAL"')
    
    st.subheader("Important Limitations")
    st.warning("The dataset contains **absolute reported crime counts** rather than **population-adjusted crime rates**.")
    st.write("Because of this, the application cannot and does not make objective claims such as:")
    st.error("❌ 'District X is the safest district.'\n❌ 'District Y is the most dangerous district.'")
    st.write("Instead, we analyze districts purely mathematically as having:")
    st.success("✅ 'comparatively higher/lower reported crime volume'\n✅ 'similar historical reported crime profiles'")
    st.write("Finally, it is important to note that historical district boundaries, population densities, and police reporting practices changed continuously between 2001 and 2012, which influences these absolute counts.")
