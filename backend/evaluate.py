import pandas as pd
from recommender import recommend


def normalize_url(url):
    if pd.isna(url):
        return ""

    url = url.strip().lower()

    # Remove domain
    url = url.replace("https://www.shl.com", "")
    url = url.replace("http://www.shl.com", "")

    # Remove optional segments
    url = url.replace("/en", "")
    url = url.replace("/solutions", "")

    # Remove trailing slash
    url = url.rstrip("/")

    return url


# Load dataset (make sure file path is correct)
train_df = pd.read_excel("data/Gen_AI Dataset.xlsx", sheet_name="Train-Set")

# Normalize ground truth URLs
train_df["Assessment_url"] = train_df["Assessment_url"].apply(normalize_url)

# Group by query
ground_truth = train_df.groupby("Query")["Assessment_url"].apply(list).to_dict()

recalls = []

for query, true_urls in ground_truth.items():

    predictions = recommend(query, top_k=10)
    predicted_urls = [normalize_url(p["url"]) for p in predictions]

    relevant_found = len(set(predicted_urls) & set(true_urls))
    recall = relevant_found / len(true_urls)

    recalls.append(recall)

    print("\n===================================")
    print("Query:", query)
    print("Recall:", recall)

    print("\nPredicted URLs:")
    for p in predicted_urls:
        print("  ", p)

    print("\nTrue URLs:")
    for t in true_urls:
        print("  ", t)


mean_recall = sum(recalls) / len(recalls)

print("\n===================================")
print("Mean Recall@10:", round(mean_recall, 4))