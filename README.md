# forensic-ai-project
Real/Fake Detection
    
---

## Project Structure

```bash
forensic-ai-project/
├── backend/
│   ├── backend.py
│   ├── model.pkl
│   ├── logistic_regression_model.pkl
│   └── requirements.txt
│
└── frontend/
    └── index.html
```

---

## Features

- Upload forensic images
- Image preprocessing
- Machine learning prediction
- Logistic Regression classification
- Web interface for user interaction

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/LeMinhPhuQN/forensic-ai-project.git
cd forensic-ai-project
```

---

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

---

## Download Trained Models

Due to GitHub file size limitations, trained models are stored on Google Drive.

Download the following files:

- [Download model.pkl](https://drive.google.com/file/d/1Ske3vEc8mfkuqFru2HlM0rYl2aIbFlKc/view?usp=drive_link)

- [Download logistic_regression_model.pkl](https://drive.google.com/file/d/1U4ETY7h7HCApJXx76PQRjkyeYcmzj9LR/view?usp=drive_link)

After downloading, place both files inside:

```bash
backend/
```

Final backend structure should look like:

```bash
backend/
├── backend.py
├── model.pkl
├── logistic_regression_model.pkl
└── requirements.txt
```

---

## Run Backend

```bash
python backend.py
```

---

## Run Frontend

Open:

```bash
frontend/index.html
```

---

## Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python

### Machine Learning
- Scikit-learn
- Joblib
- OpenCV
- NumPy

---

## Future Improvements

- Deploy full-stack application
- Improve UI/UX
- Optimize model performance
- Add real-time image analysis

---

## Run Application

### Start Backend

Open terminal:

```bash
cd backend
uvicorn backend:app --reload

## Author

**LeMinhPhuQN**