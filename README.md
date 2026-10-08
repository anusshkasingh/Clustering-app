# Customer Segmentation using Clustering (Assignment 9)

A one-file Streamlit mini project for Data Warehousing & Mining.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does
1. Two datasets, 5,000 records each (Retail Customers, Telecom Users) - satisfies "any two datasets".
2. Pick the attributes, the algorithm (K-Means or Hierarchical/Ward) and k.
3. Elbow curve + silhouette score help you choose k.
4. Clusters are shown on a 2-D PCA plot, with sizes, average profiles and a plain-English description.
5. Download the clustered data as CSV.

## Steps of the algorithm (for viva)
* Standardise attributes (z-score) so large-valued columns do not dominate distances.
* K-Means: choose k centroids, assign each point to the nearest, recompute centroids, repeat until stable.
* Hierarchical (Ward): start with every point as its own cluster and repeatedly merge the pair that increases variance least.
* Evaluate with SSE (elbow) and silhouette score.
