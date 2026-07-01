from fastapi import FastAPI
import predict as pred
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model, vectorizer = pred.load_model()

# Define the root endpoint
#@app.get("/")
#def read_root():
#    return {"message": "Fake News Detector API"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class ArticleRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(article: ArticleRequest):
    results_dict = pred.analyze_article(article.text, model, vectorizer)

    return results_dict

