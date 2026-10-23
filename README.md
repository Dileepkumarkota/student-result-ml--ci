# Student Result Prediction — MLOps Practical 6

A small Flask API that serves a trained student-result classifier, with automated model tests, a model-quality gate, a GitHub Actions release artifact, and Docker packaging.

## Features
- Reproducible training using a generated demonstration dataset.
- Flask health endpoint: `GET /`.
- Prediction endpoint: `POST /predict`.
- CI workflow trains the model, checks minimum accuracy, runs tests, and uploads a Docker-ready release artifact.
- Docker image packages the app, trained model, metrics, and Python dependencies.

> The generated data is synthetic and intended for an academic demonstration, not real student assessment.

## API input
Send JSON with all four values from 0 to 100:
```json
{
  "attendance": 90,
  "internal_marks": 85,
  "assignment_marks": 88,
  "previous_score": 80
}
```

## Run locally
```powershell
python -m pip install -r requirements.txt
python train_model.py
python quality_gate.py
python -m unittest test_ml_pipeline.py -v
python -m unittest test_app.py -v
python app.py
```

Health check: `http://localhost:5000/`

Prediction: `POST http://localhost:5000/predict`

## Practical 6 — Docker
The GitHub Actions workflow uploads the `student-result-ml-release` artifact with `Dockerfile` and `.dockerignore`. Download and extract that artifact, open a terminal in the extracted folder, then run:

```powershell
docker build -t student-result-ml:1.0 .
docker run -d --name student-result-ml-container -p 5000:5000 student-result-ml:1.0
docker ps
```

Open `http://localhost:5000/` to verify the health endpoint.

High-performance example:
```powershell
$body = @{
  attendance = 90
  internal_marks = 85
  assignment_marks = 88
  previous_score = 80
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body $body
```

Low-performance example:
```powershell
$body = @{
  attendance = 55
  internal_marks = 30
  assignment_marks = 40
  previous_score = 35
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body $body
```

Cleanup:
```powershell
docker stop student-result-ml-container
docker rm student-result-ml-container
```

Practical 6 builds and tests Docker locally. Automated Docker build/test in GitHub Actions is the next practical.
