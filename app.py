import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
import io

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/shaan098-art/MovieDash/main/movie_app_survey_synthetic.csv"
    return pd.read_csv(url)

data = load_data()

# Encode binary/categorical vars for classification/regression
@st.cache_data
def preprocess_data(df):
    df = df.copy()
    df_encoded = df.copy()
    le = LabelEncoder()
    binary_cols = df_encoded.select_dtypes(include='object').columns[df_encoded.nunique() == 2]
    for col in binary_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded

data_encoded = preprocess_data(data)

# Sidebar Navigation
tabs = st.sidebar.radio("Choose Tab", ["Data Visualization", "Classification", "Clustering", "Association Rule Mining", "Regression"])

if tabs == "Data Visualization":
    st.title("🎥 Data Visualization")
    st.write("Descriptive insights into the movie app survey data")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig = px.histogram(data, x='Age', nbins=20)
        st.plotly_chart(fig)

    with col2:
        st.subheader("Income Distribution")
        fig = px.histogram(data, x='Monthly_Income')
        st.plotly_chart(fig)

    st.subheader("Movie Genre Preferences")
    all_genres = ", ".join(data['Preferred_Genres'].dropna()).split(", ")
    genre_series = pd.Series(all_genres).value_counts()
    st.bar_chart(genre_series)

    st.subheader("Cinema Visit Frequency vs. Spend")
    fig = px.box(data, x='Cinema_Visit_Frequency', y='Spend_Per_Visit')
    st.plotly_chart(fig)

    st.subheader("AI Recommendation Interest by Age")
    fig = px.scatter(data, x='Age', y='AI_Recommendation_Interest', color='Gender')
    st.plotly_chart(fig)

    st.subheader("Preferred Time vs. Snacks Purchase")
    snack_table = pd.crosstab(data['Preferred_Time'], data['Buy_Snacks'])
    st.dataframe(snack_table)

elif tabs == "Classification":
    st.title("🤖 Classification Models")
    st.write("Apply KNN, Decision Tree, Random Forest, and GBRT classifiers")

    X = data_encoded.drop(columns=['Use_App_Regularly'])
    y = data_encoded['Use_App_Regularly']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'GBRT': GradientBoostingClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = report['accuracy'], report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']

    results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T
    st.dataframe(results_df)

    selected_model = st.selectbox("Select model for Confusion Matrix", list(models.keys()))
    model = models[selected_model]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.dataframe(pd.DataFrame(cm, columns=['Pred 0', 'Pred 1'], index=['True 0', 'True 1']))

    st.subheader("ROC Curve")
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(plt)

    st.subheader("Upload New Data for Prediction")
    uploaded = st.file_uploader("Upload CSV without target column", type=["csv"])
    if uploaded:
        new_data = pd.read_csv(uploaded)
        new_data_encoded = preprocess_data(new_data)
        preds = model.predict(new_data_encoded)
        new_data['Predicted_Use_App'] = preds
        st.dataframe(new_data)
        buffer = io.BytesIO()
        new_data.to_csv(buffer, index=False)
        st.download_button("Download Predictions", buffer.getvalue(), file_name="predictions.csv")

elif tabs == "Clustering":
    st.title("📊 KMeans Clustering")

    cluster_data = data_encoded.select_dtypes(include=['int', 'float'])
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)

    st.subheader("Elbow Chart")
    distortions = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(cluster_scaled)
        distortions.append(kmeans.inertia_)
    fig = px.line(x=list(K_range), y=distortions, labels={'x': 'Number of Clusters', 'y': 'Inertia'})
    st.plotly_chart(fig)

    k_val = st.slider("Select number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k_val, random_state=42)
    labels = kmeans.fit_predict(cluster_scaled)
    data['Cluster'] = labels

    st.subheader("Customer Personas by Cluster")
    persona = data.groupby('Cluster')[['Age', 'Monthly_Income', 'Spend_Per_Visit']].mean()
    st.dataframe(persona)

    buffer = io.BytesIO()
    data.to_csv(buffer, index=False)
    st.download_button("Download Clustered Data", buffer.getvalue(), file_name="clustered_data.csv")

elif tabs == "Association Rule Mining":
    st.title("🔗 Association Rule Mining")

    st.write("Mining rules from Preferred Genres and Snack Preferences")
    genre_df = data['Preferred_Genres'].str.get_dummies(sep=', ')
    genre_df['Buy_Snacks'] = data['Buy_Snacks'].map({'Yes': 1, 'No': 0})

    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)

    frequent_items = apriori(genre_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_conf)
    top_rules = rules.sort_values(by='confidence', ascending=False).head(10)
    st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

elif tabs == "Regression":
    st.title("📈 Regression Models")

    X = data_encoded.drop(columns=['Spend_Per_Visit'])
    y = data_encoded['Spend_Per_Visit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressors = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'DecisionTree': DecisionTreeRegressor()
    }

    st.subheader("Model Performance")
    reg_results = {}
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        reg_results[name] = round(score, 3)

    st.bar_chart(pd.Series(reg_results, name="R2 Score"))

    st.subheader("Prediction vs. Actual Spend")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
    st.plotly_chart(fig)

    st.write("The graph shows how close the predictions are to actual values, ideally along a 45° line.")
