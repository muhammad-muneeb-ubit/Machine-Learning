import joblib

model = joblib.load("../models/salary_model.pkl")

encoder = joblib.load("../models/label_encoder.pkl")

job = encoder.transform(["ML Engineer"])[0]
prediction = model.predict([[job, 6]])

print(prediction)