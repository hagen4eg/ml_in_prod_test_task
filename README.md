1. To train model please run script `train_model.py`
   
    Outputs: sumbission.csv, trained_regression_model.pkl, vectorizer.pkl
3. To start an API: `uvicorn app:app --host 0.0.0.0 --port 8000`
   
     API expects 1 input parameter with name 'text'
   
     Input example: `curl -X POST -H "Content-Type: application/json" -d '{"text": "Hello. This is the first test of this api."}' http://localhost:8000/get_text_complexity_index`
   
     Output example: {"score":-0.9324489624148711}
   
     NOTE: api requires files trained_regression_model.pkl, vectorizer.pkl (can be generated with trail_mode.py)
