from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.recommender import recommend

app = FastAPI(title="SHL Assessment Recommendation API")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recommend")
def get_recommendations(request: QueryRequest):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = recommend(request.query, top_k=request.top_k)

    return {
        "query": request.query,
        "recommendations": [
            {
                "assessment_name": r["name"],
                "assessment_url": r["url"]
            }
            for r in results
        ]
    }