from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

app = FastAPI()

iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)
class_names = ["setosa", "versicolor", "virginica"]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict")
async def predict(sl: float, sw: float, pl: float, pw: float):
    features = np.array([[sl, sw, pl, pw]])
    pred = int(model.predict(features)[0])
    return {
        "prediction": pred,
        "class_name": class_names[pred]
    }
