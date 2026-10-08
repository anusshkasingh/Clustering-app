"""Customer Segmentation using Clustering - Data Warehousing & Mining mini project (Assignment 9).

Run:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", page_icon="🔵", layout="wide")


# ----------------------------------------------------------------------------- 1. DATASETS (5,000 rows each)
@st.cache_data
def make_retail(n=5000, seed=1):
    """Dataset 1: retail shoppers drawn from 5 hidden groups."""
    rng = np.random.default_rng(seed)
    #        age  income spend visits basket   (means)        group weights
    g = np.array([[24, 35, 75, 9, 900],      # young trendy spenders
                  [45, 95, 80, 7, 2600],     # rich big spenders
                  [50, 90, 20, 2, 600],      # rich but saving
                  [30, 30, 25, 3, 400],      # budget shoppers
                  [58, 55, 50, 5, 1400]])    # steady seniors
    sd = np.array([4, 8, 8, 1.5, 250])
    z = rng.choice(5, n, p=[.22, .18, .15, .27, .18])
    x = g[z] + rng.normal(0, 1, (n, 5)) * sd
    df = pd.DataFrame(x, columns=["age", "annual_income_k", "spending_score", "visits_per_month", "avg_basket_value"])
    df = df.clip(lower=[18, 10, 1, 1, 100], upper=[75, 160, 100, 15, 5000]).round(1)
    df.insert(0, "customer_id", np.arange(1, n + 1))
    return df


@st.cache_data
def make_telecom(n=5000, seed=2):
    """Dataset 2: mobile-plan users drawn from 4 hidden groups."""
    rng = np.random.default_rng(seed)
    #        minutes data_gb bill tenure support_calls
    g = np.array([[900, 3, 450, 40, 1],      # voice-heavy
                  [250, 28, 800, 24, 1],     # data-heavy
                  [150, 4, 250, 8, 4],       # new & unhappy
                  [500, 12, 600, 60, 0.5]])  # loyal all-rounders
    sd = np.array([120, 4, 80, 8, 1])
    z = rng.choice(4, n, p=[.28, .25, .22, .25])
    x = g[z] + rng.normal(0, 1, (n, 5)) * sd
    df = pd.DataFrame(x, columns=["monthly_minutes", "data_gb", "monthly_bill", "tenure_months", "support_calls"])
    df = df.clip(lower=0).round(1)
    df.insert(0, "user_id", np.arange(1, n + 1))
    return df


DATASETS = {"Retail Customers": make_retail, "Telecom Users": make_telecom}


# ----------------------------------------------------------------------------- 2. CLUSTERING FUNCTIONS
def scale(df, cols):
    scaler = StandardScaler()
    Z = scaler.fit_transform(df[cols])
    return Z, scaler



def predict_cluster(scaler, kmeans_model, input_values):
    """Scale a single input row and return the predicted cluster label."""
    arr = np.array(input_values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return int(kmeans_model.predict(arr_scaled)[0])


def cluster(Z, k, method):
    if method == "K-Means":
        m = KMeans(n_clusters=k, n_init=10, random_state=1).fit(Z)
        return m.labels_, m
    m = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z)
    return m.labels_, None


def describe(profile, overall):
    """Plain-English label per cluster: attributes more than 0.7 SD above/below the overall mean."""
    out = {}
    for c, row in profile.iterrows():
        words = []
        for col in overall.columns:
            z = (row[col] - overall[col].mean()) / overall[col].std()
            if z > 0.7:
                words.append(f"High {col}")
            elif z < -0.7:
                words.append(f"Low {col}")
        out[c] = ", ".join(words) if words else "Average on all attributes"
    return out


# ----------------------------------------------------------------------------- 3. USER INTERFACE
st.title("🔵 Customer Segmentation using Clustering")
st.caption("Explore how customers naturally group together based on their behaviour and attributes using K-Means and Hierarchical clustering.")

with st.sidebar:
    st.header("Settings")
    name = st.selectbox("1. Choose dataset", list(DATASETS))
    df = DATASETS[name]()
    id_col = df.columns[0]
    numeric = [c for c in df.columns if c != id_col]
    feats = st.multiselect("2. Attributes to cluster on", numeric, default=numeric)
    method = st.radio("3. Algorithm", ["K-Means", "Hierarchical"])
    k = st.slider("4. Number of clusters (k)", 2, 9, 4)
    st.caption("Hierarchical clustering uses a random sample of 2,000 rows because it needs O(n²) memory.")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Data", "📈 Scores & Prediction", "🎯 Clusters", "📊 Insights"])

with tab1:
    a, b, c = st.columns(3)
    a.metric("Records", f"{len(df):,}")
    b.metric("Attributes", len(numeric))
    c.metric("Missing values", int(df.isna().sum().sum()))
    st.dataframe(df.head(100))
    st.caption("Showing the first 100 rows of the dataset — scroll right to see all attributes.")
    st.subheader("Summary statistics")
    st.dataframe(df[numeric].describe().T.round(2))
    st.caption("Each row is one attribute. The mean and std give a quick sense of the data's range and spread — large std values mean that attribute varies a lot across customers.")

if len(feats) < 2:
    st.warning("Select at least two attributes in the sidebar.")
    st.stop()

data = df if method == "K-Means" else df.sample(2000, random_state=1)
Z, scaler = scale(data, feats)

labels, fitted_model = cluster(Z, k, method)
out = data.copy()
out["cluster"] = labels
sil = silhouette_score(Z, labels, sample_size=min(3000, len(Z)), random_state=1)
profile = out.groupby("cluster")[feats].mean().round(2)
profile.insert(0, "size", out.groupby("cluster").size())
names = describe(profile[feats], out[feats])

with tab2:
    # ---- Clustering quality scores for chosen k ----
    st.subheader(f"Clustering quality scores  (k = {k}, {method})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette score", f"{sil:.3f}",
              help="Range −1 to 1. Closer to 1 = clusters well separated.")
    m2.metric("Davies-Bouldin index", f"{davies_bouldin_score(Z, labels):.3f}",
              help="Lower is better. 0 = perfect separation.")
    m3.metric("Calinski-Harabasz score", f"{calinski_harabasz_score(Z, labels):.1f}",
              help="Higher is better. Ratio of between- to within-cluster variance.")
    m4.metric("SSE (Inertia)", f"{(fitted_model.inertia_ if fitted_model else float('nan')):.1f}" if fitted_model else "N/A (Hierarchical)",
              help="Sum of squared distances to cluster centres. Lower = tighter clusters.")

    # ---- Score charts across k values ----
    st.divider()
    st.subheader("How scores change across k values")
    st.caption("These charts help you pick the best number of clusters.")

    score_rows = []
    for ki in range(2, 10):
        km = KMeans(n_clusters=ki, n_init=5, random_state=1).fit(Z)
        score_rows.append({
            "k": ki,
            "Silhouette": silhouette_score(Z, km.labels_, sample_size=2000, random_state=1),
            "Davies-Bouldin": davies_bouldin_score(Z, km.labels_),
            "Calinski-Harabasz": calinski_harabasz_score(Z, km.labels_),
            "SSE (Inertia)": km.inertia_,
        })
    score_df = pd.DataFrame(score_rows).set_index("k")

    ca, cb = st.columns(2)
    ca.markdown("**Silhouette score** *(higher = better)*")
    ca.line_chart(score_df["Silhouette"])
    cb.markdown("**Davies-Bouldin index** *(lower = better)*")
    cb.line_chart(score_df["Davies-Bouldin"])

    cc, cd = st.columns(2)
    cc.markdown("**Calinski-Harabasz score** *(higher = better)*")
    cc.line_chart(score_df["Calinski-Harabasz"])
    cd.markdown("**SSE / Inertia (Elbow)** *(look for the bend)*")
    cd.line_chart(score_df["SSE (Inertia)"])

    best_k = int(score_df["Silhouette"].idxmax())
    st.success(f"Best silhouette at **k = {best_k}** ({score_df['Silhouette'].max():.3f}). You chose k = {k}.")
    st.dataframe(score_df.round(3))
    st.caption("Each row is a different value of k. Compare across rows to find where scores peak (silhouette, Calinski-Harabasz) or dip (Davies-Bouldin, SSE) — that's your ideal number of clusters.")

    # ---- Cluster prediction ----
    st.divider()
    st.subheader("🔮 Predict cluster for a new data point")
    if method != "K-Means":
        st.info("Prediction requires K-Means (Hierarchical has no predict method). Switch the algorithm in the sidebar.")
    else:
        st.caption("Adjust the sliders to describe a new data point, then click **Predict**.")
        col_inputs = st.columns(len(feats))
        input_vals = []
        for i, feat in enumerate(feats):
            col_mean = float(data[feat].mean())
            col_min  = float(data[feat].min())
            col_max  = float(data[feat].max())
            val = col_inputs[i].number_input(
                feat,
                min_value=col_min,
                max_value=col_max,
                value=round(col_mean, 1),
                step=0.1,
            )
            input_vals.append(val)

        if st.button("Predict cluster", type="primary"):
            predicted = predict_cluster(scaler, fitted_model, input_vals)
            cluster_desc = names.get(predicted, "—")
            st.success(f"This data point belongs to **Cluster {predicted}** — *{cluster_desc}*")
            st.markdown("**Cluster profile (average values):**")
            st.dataframe(profile.loc[[predicted]])
            st.caption("These are the average attribute values for everyone in this cluster — use them to understand what kind of customer this data point resembles.")


with tab3:
    a, b, c = st.columns(3)
    a.metric("Records clustered", f"{len(out):,}")
    b.metric("Clusters", k)
    c.metric("Silhouette score", f"{sil:.3f}", help="Closer to 1 = well separated clusters")
    pcs = PCA(2, random_state=1).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="tab10", s=8, alpha=.7)
    ax.set_xlabel("Principal component 1"); ax.set_ylabel("Principal component 2")
    ax.set_title(f"{method} clusters (PCA view)")
    ax.legend(*sc.legend_elements(), title="cluster", loc="best")
    st.pyplot(fig); plt.close(fig)
    st.subheader("Cluster sizes")
    st.bar_chart(out.groupby("cluster").size())

with tab4:
    st.subheader("Cluster profiles (average of each attribute)")
    st.dataframe(profile)
    st.caption("Each row is one cluster. The numbers show the average value of each attribute for customers in that cluster — higher or lower values reveal what makes each group distinct.")
    st.subheader("What each cluster means")
    for c in profile.index:
        st.markdown(f"**Cluster {c}** ({int(profile.loc[c, 'size']):,} records, {profile.loc[c, 'size'] / len(out):.0%}) — {names[c]}")
    st.subheader("Compare clusters")
    attr = st.selectbox("Attribute", feats)
    st.bar_chart(profile[attr])
    st.download_button("⬇️ Download clustered data (CSV)", out.to_csv(index=False), "clustered_data.csv")
