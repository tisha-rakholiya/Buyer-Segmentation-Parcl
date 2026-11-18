# Buyer Segmentation and Investment Profiling - Parcl Co. Limited
# ----------------------------------------------------------------
# Put clients.csv and properties.csv in the same folder as this file, then run:
#     python segmentation_analysis.py
# Everything it creates is saved in the "outputs" folder.

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import mannwhitneyu, chi2_contingency

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

CLIENTS_FILE = "clients.csv"
PROPERTIES_FILE = "properties.csv"
OUT = "outputs"
CHOSEN_K = 5            # number of clusters (see the explanation in STEP 5 below)
os.makedirs(OUT, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()


# =====================================================
# STEP 1: LOAD AND CLEAN
# =====================================================
print("STEP 1: Loading and cleaning data")

clients = pd.read_csv(CLIENTS_FILE)
properties = pd.read_csv(PROPERTIES_FILE)
print("clients:", clients.shape, "| properties:", properties.shape)
print("\nMissing values in clients:\n", clients.isnull().sum())
print("Missing values in properties:\n", properties.isnull().sum())
print("(client_ref is empty for 'Available' units because nobody has bought them yet)")

# remove duplicates
print("\nDuplicate clients:", clients.duplicated("client_id").sum())
print("Duplicate listings:", properties.duplicated("listing_id").sum())
clients = clients.drop_duplicates("client_id")
properties = properties.drop_duplicates("listing_id")

# make text labels consistent
for col in ["client_type", "acquisition_purpose", "referral_channel", "gender", "country", "region"]:
    clients[col] = clients[col].str.strip()
for col in ["client_type", "acquisition_purpose", "referral_channel"]:
    clients[col] = clients[col].str.title()
clients["gender"] = clients["gender"].map({"M": "Male", "F": "Female"}).fillna("Unknown")

# fill any missing values (there are none in this dataset, but the brief asks for it)
for col in clients.columns:
    if clients[col].dtype == "object" or str(clients[col].dtype) == "str":
        clients[col] = clients[col].fillna("Unknown")
    else:
        clients[col] = clients[col].fillna(clients[col].median())

# age: date_of_birth has two formats (05-11-1968 and 11/26/1962) so day/month can be confusing.
# The year is always the last 4 digits, so we calculate age from the birth year.
clients["birth_year"] = clients["date_of_birth"].str.extract(r"(\d{4})$")[0].astype(int)
clients["age"] = 2025 - clients["birth_year"]

clients["loan_num"] = (clients["loan_applied"] == "Yes").astype(int)

# properties: "$300,385.62" -> 300385.62, and real dates
properties["price"] = properties["sale_price"].str.replace(r"[$,]", "", regex=True).astype(float)
properties["date"] = pd.to_datetime(properties["transaction_date"], format="%m-%d-%Y")

# purchase behaviour of every client (only Sold units belong to a client)
sold = properties[properties["listing_status"] == "Sold"]
buyer_stats = sold.groupby("client_ref").agg(
    units_bought=("listing_id", "count"),
    total_spend=("price", "sum"),
    avg_unit_price=("price", "mean"),
    office_share=("unit_category", lambda x: (x == "Office").mean()),
).reset_index().rename(columns={"client_ref": "client_id"})

df = clients.merge(buyer_stats, on="client_id", how="left")
purchase_cols = ["units_bought", "total_spend", "avg_unit_price", "office_share"]
print("\nClients with no purchase:", int(df["units_bought"].isnull().sum()))
df[purchase_cols] = df[purchase_cols].fillna(0)
print("Final table:", df.shape)

# codes used later for the maps
iso3 = {"USA": "USA", "UK": "GBR", "Canada": "CAN", "Germany": "DEU", "France": "FRA", "Belgium": "BEL",
        "Mexico": "MEX", "Australia": "AUS", "Russia": "RUS", "Denmark": "DNK"}
states = {"Arizona": "AZ", "California": "CA", "Colorado": "CO", "Florida": "FL", "Georgia": "GA", "Kansas": "KS",
          "Nevada": "NV", "New York": "NY", "Ohio": "OH", "Oregon": "OR", "Texas": "TX", "Utah": "UT",
          "Virginia": "VA", "Washington": "WA", "Wyoming": "WY"}
df["country_iso3"] = df["country"].map(iso3)
df["state_code"] = np.where(df["country"] == "USA", df["region"].map(states), None)


# =====================================================
# EDA (charts for the research paper)
# =====================================================
print("\nEDA: saving charts")

plt.figure(figsize=(6, 4))
sns.histplot(df["age"], bins=20)
plt.title("Age of clients")
save("eda_age.png")

plt.figure(figsize=(6, 4))
sns.countplot(x="satisfaction_score", data=df)
plt.title("Satisfaction score")
save("eda_satisfaction.png")

plt.figure(figsize=(7, 4))
sns.countplot(y="country", data=df, order=df["country"].value_counts().index)
plt.title("Clients by country")
save("eda_country.png")

plt.figure(figsize=(6, 4))
sns.countplot(x="referral_channel", hue="acquisition_purpose", data=df)
plt.title("Referral channel and purpose")
save("eda_referral_purpose.png")

plt.figure(figsize=(6, 4))
sns.countplot(x="client_type", hue="loan_applied", data=df)
plt.title("Client type and loan")
save("eda_type_loan.png")

plt.figure(figsize=(6, 4))
sns.histplot(properties["price"], bins=30)
plt.title("Sale price of units")
save("eda_price.png")

plt.figure(figsize=(6, 4))
sns.countplot(x="units_bought", data=df)
plt.title("Units bought per client")
save("eda_units_per_client.png")

monthly = sold.groupby(sold["date"].dt.to_period("M")).size()
plt.figure(figsize=(8, 4))
monthly.plot()
plt.title("Units sold per month")
plt.xlabel("Month")
save("eda_monthly_sales.png")

print("Average units bought:  Home =", round(df[df.acquisition_purpose == "Home"]["units_bought"].mean(), 2),
      "| Investment =", round(df[df.acquisition_purpose == "Investment"]["units_bought"].mean(), 2))
print("Average units bought:  Individual =", round(df[df.client_type == "Individual"]["units_bought"].mean(), 2),
      "| Company =", round(df[df.client_type == "Company"]["units_bought"].mean(), 2))


# =====================================================
# STEP 2: ENCODING
# =====================================================
print("\nSTEP 2: Encoding")

# One-hot encoding for categories with few values.
# 'region' has 57 different values, so it is only label-encoded (for reference) and used in the maps.
one_hot_cols = ["client_type", "acquisition_purpose", "referral_channel", "country"]
encoded = pd.get_dummies(df[one_hot_cols], dtype=float)
df["country_code"] = LabelEncoder().fit_transform(df["country"])
df["region_code"] = LabelEncoder().fit_transform(df["region"])
print("One-hot columns created:", encoded.shape[1])


# =====================================================
# STEP 3: SCALING
# =====================================================
print("\nSTEP 3: Scaling")

num_cols = ["age", "satisfaction_score", "units_bought", "avg_unit_price", "office_share"]
scaled = StandardScaler().fit_transform(df[num_cols])
X = np.hstack([scaled, df[["loan_num"]].values, encoded.values])
print("Feature matrix:", X.shape)


# =====================================================
# STEP 4 & 5: HOW MANY CLUSTERS? (Elbow + Silhouette)
# =====================================================
print("\nSTEP 5: Elbow method and silhouette score")

ks = list(range(2, 9))
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    print(f"K={k}  inertia={km.inertia_:.0f}  silhouette={sil[-1]:.3f}")

# The silhouette scores are almost the same for every K (about 0.11), so the numbers alone cannot decide.
# We use K=5 because: (1) it has the best silhouette among K=3,4,5, and
# (2) it separates a small group of "bulk buyers" that K=4 hides inside a bigger group.
best_k = CHOSEN_K
print("Chosen K:", best_k)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, marker="o")
ax[0].set_title("Elbow method")
ax[0].set_xlabel("K")
ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil, marker="o", color="green")
ax[1].set_title("Silhouette score")
ax[1].set_xlabel("K")
save("elbow_silhouette.png")

print("\nSTEP 4: K-Means and Hierarchical clustering")

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
df["cluster"] = kmeans.labels_
print("K-Means silhouette:", round(silhouette_score(X, kmeans.labels_), 3))

hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(X)
print("Hierarchical silhouette:", round(silhouette_score(X, hier.labels_), 3))
print("Agreement between the two (adjusted rand index):", round(adjusted_rand_score(kmeans.labels_, hier.labels_), 3))

plt.figure(figsize=(10, 4))
dendrogram(linkage(X, method="ward"), truncate_mode="lastp", p=20, no_labels=True)
plt.title("Hierarchical clustering dendrogram")
save("dendrogram.png")


# =====================================================
# STEP 6: UNDERSTAND EACH CLUSTER
# =====================================================
print("\nSTEP 6: Cluster profiles")

profile = df.groupby("cluster").agg(
    clients=("client_id", "count"),
    avg_age=("age", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
    avg_units=("units_bought", "mean"),
    avg_unit_price=("avg_unit_price", "mean"),
    avg_total_spend=("total_spend", "mean"),
    office_share=("office_share", "mean"),
    loan_rate=("loan_num", "mean"),
    investment_rate=("acquisition_purpose", lambda x: (x == "Investment").mean()),
    company_rate=("client_type", lambda x: (x == "Company").mean()),
    usa_rate=("country", lambda x: (x == "USA").mean()),
).round(2)
profile["share_pct"] = (profile["clients"] / len(df) * 100).round(1)

# Give every cluster a readable name: find the 2 things that make it most different from all clients.
traits = {   # column: (word if higher than average, word if lower than average)
    "age": ("Older", "Younger"),
    "satisfaction_score": ("Satisfied", "Dissatisfied"),
    "units_bought": ("Bulk", None),     # only "Bulk" is used, low units is not meaningful
    "avg_unit_price": ("Premium", "Budget"),
    "office_share": ("Office-Focused", "Apartment-Focused"),
    "loan_num": ("Loan-Financed", "No-Loan"),
}
names = {}
for c, group in df.groupby("cluster"):
    found = []
    for col in traits:
        diff = (group[col].mean() - df[col].mean()) / df[col].std()
        word = traits[col][0] if diff > 0 else traits[col][1]
        if word is not None and abs(diff) >= 0.3:
            found.append((abs(diff), word))
    found.sort(reverse=True)
    words = [w for _, w in found[:2]]
    names[c] = (" & ".join(words) if words else "Average") + " Buyers"

profile.insert(0, "segment", profile.index.map(names))
df["segment"] = df["cluster"].map(names)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(profile.to_string())

profile.to_csv(os.path.join(OUT, "cluster_profile.csv"))

# the dashboard file: no names of clients are saved
keep = ["client_id", "client_type", "gender", "country", "country_iso3", "region", "state_code",
        "acquisition_purpose", "referral_channel", "satisfaction_score", "loan_applied", "loan_num",
        "age", "units_bought", "total_spend", "avg_unit_price", "office_share", "cluster", "segment"]
df[keep].to_csv(os.path.join(OUT, "segmented_clients.csv"), index=False)


# =====================================================
# CHARTS FOR THE PAPER
# =====================================================
order = list(profile["segment"])

plt.figure(figsize=(8, 4))
sns.countplot(y="segment", data=df, order=order)
plt.title("Number of clients in each segment")
save("cluster_distribution.png")

plt.figure(figsize=(8, 4))
pd.crosstab(df["segment"], df["acquisition_purpose"], normalize="index").mul(100).loc[order].plot(kind="barh", stacked=True)
plt.title("Acquisition purpose in each segment (%)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
save("purpose_by_cluster.png")

plt.figure(figsize=(8, 4))
sns.barplot(y="segment", x="loan_num", data=df, order=order, errorbar=None)
plt.title("Loan rate in each segment")
save("loan_by_cluster.png")

plt.figure(figsize=(8, 4))
sns.barplot(y="segment", x="total_spend", data=df, order=order, errorbar=None)
plt.title("Average total spend per client")
save("spend_by_cluster.png")

points = PCA(n_components=2, random_state=42).fit_transform(X)
plt.figure(figsize=(7, 5))
sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=df["segment"], s=15)
plt.title("Clients in 2D (PCA)")
save("clusters_pca.png")


# =====================================================
# STEP 7: CHECKS (statistical tests and stability)
# =====================================================
print("\nSTEP 7: Statistical checks")

# (a) Do the stated groups really behave differently? (Mann-Whitney U test, p < 0.05 means a real difference)
rows = []
for col, a, b in [("acquisition_purpose", "Home", "Investment"), ("client_type", "Individual", "Company"), ("loan_applied", "Yes", "No")]:
    for target in ["units_bought", "total_spend", "satisfaction_score"]:
        x = df[df[col] == a][target]
        y = df[df[col] == b][target]
        rows.append({"test": f"{a} vs {b}", "variable": target, "mean_first": round(x.mean(), 2),
                     "mean_second": round(y.mean(), 2), "p_value": round(mannwhitneyu(x, y).pvalue, 4)})
tests = pd.DataFrame(rows)
print(tests.to_string(index=False))

# (b) Are the segments linked to purpose / client type? (chi-square test)
for col in ["acquisition_purpose", "client_type"]:
    chi_p = chi2_contingency(pd.crosstab(df["segment"], df[col]))[1]
    print(f"Chi-square, segment vs {col}: p = {chi_p:.4f}")
    tests.loc[len(tests)] = {"test": f"segment vs {col}", "variable": "chi-square", "p_value": round(chi_p, 4)}

# (c) Are the clusters stable? Run K-Means with 10 different random seeds and compare with our result
aris = []
for seed in range(10):
    other = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X).labels_
    aris.append(adjusted_rand_score(df["cluster"], other))
print(f"Stability: average agreement with other random seeds (ARI) = {np.mean(aris):.3f} (min {np.min(aris):.3f})")
tests.loc[len(tests)] = {"test": "stability over 10 seeds", "variable": "ARI mean", "p_value": round(float(np.mean(aris)), 3)}

tests.to_csv(os.path.join(OUT, "statistical_tests.csv"), index=False)

print("\nDONE. Check the 'outputs' folder.")
print("Clients:", len(df), "| K:", best_k, "| segments:", list(names.values()))
