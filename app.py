from fastapi import FastAPI
import predict as pred
from pydantic import BaseModel

app = FastAPI()

model, vectorizer = pred.load_model()

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Fake News Detector API"}

class ArticleRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(article: ArticleRequest):
    results_dict = pred.analyze_article(article.text, model, vectorizer)

    return results_dict

