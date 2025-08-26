Adult Income Prediction Project

A comprehensive machine learning project that predicts whether an individual's income exceeds $50K/year based on census data from the UCI Machine Learning Repository.
📁Project Structure
text

project/
└── practical/
    ├── task1.py          # Data loading and exploratory data analysis
    ├── task2.py          # Preprocessing pipeline
    ├── task3.py          # Model training and hyperparameter tuning
    ├── task4.py          # Logistic Regression from scratch
    ├── task5.py          # FastAPI deployment
    ├── task6.py          # Model explainability and robustness testing
    ├── requirements.txt  # Python dependencies
    └── README.md         # This file

📊 Dataset

Source: UCI Machine Learning Repository - Adult Dataset
URL: https://archive.ics.uci.edu/ml/datasets/adult
Samples: 32,561
Features: 14 demographic and employment-related features
Target: Binary classification (>50K or ≤50K)
🚀 Features

    Data Exploration & Visualization: Comprehensive EDA with correlation analysis

    Data Preprocessing: Handling missing values, feature encoding, and scaling

    Machine Learning Models: Logistic Regression and Random Forest classifiers

    Hyperparameter Tuning: Randomized search for optimal parameters

    Custom Implementation: Logistic Regression from scratch

    Model Explainability: SHAP analysis for feature importance

    API Deployment: FastAPI endpoint for model predictions

    Robustness Testing: Noise injection and performance evaluation

🛠 Installation & Setup

    Navigate to the project directory
    bash

cd project/practical

Create a virtual environment (recommended)
bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
bash

    pip install -r requirements.txt

🏃‍♂️ How to Run
Running from the Project Root
bash

# Navigate to the practical folder first
cd project/practical

# Run tasks in sequence
python task1.py  # Data loading and EDA
python task2.py  # Preprocessing pipeline
python task3.py  # Model training and tuning
python task4.py  # Scratch implementation (optional)
python task5.py  # API server (after model training)
python task6.py  # Explainability and robustness

Or Using Full Paths
bash

# From any location
python project/practical/task1.py
python project/practical/task2.py
# ... etc

📋 Task Details
Task 1: Data Loading and EDA
bash

cd project/practical
python task1.py

    Purpose: Initial data exploration and visualization

    Outputs: Dataset statistics, correlation heatmaps, missing value analysis

    Files Created: Dataset download (adult.data), various plots

Task 2: Preprocessing Pipeline
bash

python task2.py

    Purpose: Data cleaning, encoding, and scaling

    Dependencies: Requires Task 1 to be run first

    Output: Preprocessing pipeline validation

Task 3: Model Training
bash

python task3.py

    Purpose: Train and optimize machine learning models

    Dependencies: Requires Tasks 1-2

    Outputs: Trained models, performance metrics, ROC curve, saved model file (adult_rf_model.joblib)

Task 4: Scratch Implementation
bash

python task4.py

    Purpose: Educational implementation of Logistic Regression

    Note: Optional task, uses synthetic data

Task 5: API Deployment
bash

python task5.py

    Purpose: Create REST API for model predictions

    Dependencies: Requires trained model from Task 3

    API: FastAPI server on http://localhost:8000

Task 6: Explainability and Robustness
bash

python task6.py

    Purpose: Model interpretation and robustness testing

    Dependencies: Requires trained model and test data

    Outputs: SHAP plots, robustness analysis

🌐 API Usage

After running Task 5, access the API at http://localhost:8000
Health Check
bash

curl http://localhost:8000/health

Make a Prediction
bash

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "workclass": "Private",
    "fnlwgt": 100000,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 5000,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
  }'

📊 Expected Results

    Task 1: Data insights and visualizations

    Task 2: Preprocessing pipeline validation

    Task 3: Model accuracy ~85-87%, F1 score ~0.67-0.72

    Task 4: Educational implementation with ~80-85% accuracy

    Task 5: Functional API endpoint

    Task 6: Feature importance analysis and robustness report

⚠️ Important Notes

    Execution Order: Run tasks 1-3 sequentially for proper data flow

    File Dependencies:

        Task 2 requires data from Task 1

        Task 3 requires preprocessing from Task 2

        Task 5 requires the saved model from Task 3

        Task 6 requires model and test data from previous tasks

    Internet Connection: Required for Task 1 to download dataset

    File Paths: All tasks assume execution from the practical/ directory

🔧 Troubleshooting

Common Issues:

    ModuleNotFoundError: Ensure you're in the practical/ directory and dependencies are installed

    FileNotFoundError: Run tasks in correct order (1 → 2 → 3 → 5 → 6)

    API not starting: Check if port 8000 is available

Solutions:
bash

# Reinstall dependencies
pip install -r requirements.txt

# Check current directory
pwd  # Should be project/practical/

# Force re-download dataset
rm adult.data  # Then re-run task1.py

📝 License

This project is for educational purposes. The dataset is from the UCI Machine Learning Repository.
🤝 Support

For issues or questions:

    Ensure all dependencies are installed

    Verify execution order is correct

    Check you're running from the practical/ directory

Note: This project demonstrates a complete ML workflow from data exploration to deployment. Follow the task order for best results!

