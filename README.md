# Fresh/Rotten Fruit Classifier

Small web app to classify fruit quality (fresh/rotten) using the provided Keras model.

## Setup

1) Create a virtual environment (optional)
2) Install dependencies:

```
pip install -r requirements.txt
```

## Run

```
python app.py
```

Open http://localhost:8000

## Notes
- If you do not have GPU, TensorFlow will use CPU by default.
- For video classification, the app samples frames and uses majority vote.
