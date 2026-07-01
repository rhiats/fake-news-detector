from fastapi import FastAPI
import predict as pred

app = FastAPI()

model, vectorizer = pred.load_model()



