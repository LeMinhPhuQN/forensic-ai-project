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

### 1. Start Backend

Open terminal:

```bash
cd backend
uvicorn backend:app --reload
```

---

### 2. Configure Frontend API

Before running the frontend, open:

```bash
frontend/index.html
```

Go to:

```bash
Line 1170
```
Replace the API URL with your backend URL.

---

### 3. Run Frontend

You can run the frontend using:

#### Option 1 — VS Code Live Server (Recommended)

Install:

- Live Server extension in VS Code

Then:

- Right click `index.html`
- Select:

```text
Open with Live Server
```

Frontend will run automatically in your browser.

---

#### Option 2 — Open HTML Directly

Open:

```bash
frontend/index.html
```

directly in your browser.

---

## Author

**LeMinhPhuQN**