import pandas as pd
from recommender import recommend

# Load test queries
test_df = pd.read_excel("data/Gen_AI Dataset.xlsx", sheet_name="Test-Set")

rows = []

for query in test_df["Query"]:

    predictions = recommend(query, top_k=10)

    for p in predictions:
        rows.append({
            "Query": query,
            "Assessment_url": p["url"]
        })

submission_df = pd.DataFrame(rows)

submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")