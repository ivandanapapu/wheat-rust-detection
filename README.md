# Wheat Rust Detection: Binary Image Classifier

A complete ML project for detecting whether a wheat leaf image is Rust affected or Healthy using a CNN binary image classifier.

## Project Parts
- `ml/` - CNN model training, augmentation, metrics and false-negative analysis
- `backend/` - FastAPI prediction API
- `frontend/` - React + Vite attractive UI for image upload and result display

## Dataset
Kaggle: https://www.kaggle.com/datasets/admin00700/wheat-leaf-dataset

Expected dataset format after download:

```text
dataset/
  healthy/
    image1.jpg
  rust/
    image2.jpg
```

If the Kaggle dataset has more classes, keep all rust/disease images under `rust` and healthy images under `healthy` for binary classification.

## Train Model
```bash
cd ml
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python train.py
```

The trained model will be saved as:

```text
ml/models/wheat_rust_cnn.keras
```

Copy this model into backend:

```bash
copy ml\models\wheat_rust_cnn.keras backend\model\wheat_rust_cnn.keras
```

## Run Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

## Run Frontend
```bash
cd frontend
npm install
npm run dev
```

Create `.env` inside frontend:

```text
VITE_API_URL=http://127.0.0.1:8000
```

## Deploy
- Frontend: Vercel
- Backend: Render/Railway
- Set Vercel environment variable:

```text
VITE_API_URL=https://your-backend-url.onrender.com
```
