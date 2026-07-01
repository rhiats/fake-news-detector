from fastapi import FastAPI
import predict as pred

app = FastAPI()

model, vectorizer = pred.load_model()

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Fake News Detector API"}

