# Amazon Dynamic Pricing Impact Analysis
## Machine Learning Capstone Project - CRISP-DM Methodology

### Project Overview

This project applies the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to predict customer recommendation likelihood based on demographics, shopping behavior, and perception of Amazon's dynamic pricing strategy.

**Research Question:** Can we predict whether a customer is likely to recommend Amazon based on their demographics, shopping behavior, and perception of Amazon's dynamic pricing strategy?

**Problem Type:** Multi-class Classification

**Target Variable:** `Likely_to_Recommend_Amazon_Based_on_Pricing`
- Classes: Highly Likely, Likely, Unlikely, Highly Unlikely

---

## Dataset Information

**Dataset Name:** Amazon Dynamic Pricing Survey Data

**Source:** Provided dataset in project folder (`amazon_dynamic_pricing_survey.csv`)

**Size:** 5,000 customer records

**Features:**
- **Demographics:** Age, Gender, Location, Annual_Income
- **Behavior:** Browsing_Time_per_Week_Hours, Purchase_Frequency_Per_Month
- **Perceptions:** 
  - Impact_of_Dynamic_Pricing_on_Purchase
  - Perception_of_Amazon_Revenue_Growth_due_to_Dynamic_Pricing
  - Perception_of_Competition_in_Amazon_Marketplace

**Note:** The dataset file is not included in this repository due to size constraints. Please ensure the `amazon_dynamic_pricing_survey.csv` file is in the project root directory before running the notebook.

---

## Project Structure

```
AI_Project/
│
├── amazon_dynamic_pricing_analysis.ipynb  # Main Jupyter notebook with complete analysis
├── README.md                               # This file
├── requirements.txt                        # Python package dependencies
├── .gitignore                              # Git ignore file
├── models/                                 # Saved model files (generated after running notebook)
│   ├── best_model.pkl
│   ├── label_encoders.pkl
│   ├── target_encoder.pkl
│   ├── feature_names.pkl
│   └── scaler.pkl (if Neural Network is best model)
├── dashboard.py                            # Interactive dashboard application
└── amazon_dynamic_pricing_survey.csv       # Dataset (not in repo, add locally)
```

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd AI_Project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Dataset

Ensure the `amazon_dynamic_pricing_survey.csv` file is in the project root directory.

---

## Usage

### Running the Main Analysis

1. Open Jupyter Notebook:
   ```bash
   jupyter notebook amazon_dynamic_pricing_analysis.ipynb
   ```

2. Run all cells sequentially to:
   - Load and explore the data
   - Perform data preparation and feature engineering
   - Conduct exploratory data analysis
   - Train and compare multiple ML models
   - Evaluate model performance
   - Save the best model for deployment

### Running the Interactive Dashboard

1. Ensure you've run the notebook first to generate the model files in the `models/` directory.

2. Run the dashboard:
   ```bash
   python dashboard.py
   ```

3. Open your web browser and navigate to `http://localhost:8050`

4. Use the interactive interface to:
   - Input customer demographics and behavior
   - Select perception values
   - Get real-time predictions of recommendation likelihood

---

## CRISP-DM Phases

### Phase 1: Business & Data Understanding
- Defined research question and ML problem type
- Analyzed dataset structure and characteristics
- Identified target variable and features

### Phase 2: Data Preparation
- Data cleaning and validation
- Feature engineering (income categories, age groups, etc.)
- Categorical encoding and numerical scaling
- Train/test split (80/20)

### Phase 3: Exploratory Data Analysis (EDA)
- Distribution analysis of all features
- Correlation analysis
- Relationship exploration between features and target
- Visualization of key patterns

### Phase 4: Modeling
Trained and compared 4 different models:
1. **Decision Tree Classifier** - Interpretable baseline model
2. **Random Forest Classifier** - Ensemble method, reduces overfitting
3. **Gradient Boosting Classifier** - Strong performance on complex patterns
4. **Neural Network (MLP)** - Captures non-linear relationships

### Phase 5: Evaluation
- Accuracy and F1-score metrics
- Classification reports for each class
- Confusion matrices
- 5-fold cross-validation
- Model comparison and selection

### Phase 6: Deployment
- Model serialization for production use
- Interactive dashboard for predictions
- Prediction function for new data
- Business insights and recommendations

---

## Model Performance

The best performing model will be automatically selected based on test accuracy. Results are displayed in the notebook after training.

**Evaluation Metrics:**
- Accuracy Score
- F1-Score (Weighted)
- Precision, Recall per class
- Cross-Validation Scores

---

## Key Findings

1. **Feature Importance:** Customer perceptions of dynamic pricing impact, revenue growth, and competition are the strongest predictors of recommendation likelihood.

2. **Behavioral Patterns:** Higher purchase frequency and browsing time correlate with positive recommendations.

3. **Demographics:** Age and income show moderate influence on recommendation likelihood.

4. **Business Value:** The model can help Amazon identify customer segments likely to recommend the platform and develop targeted strategies.

---

## Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning models and evaluation
- **Joblib** - Model serialization
- **Dash/Plotly** - Interactive dashboard (for data product)

---

## Team Members

[Pooja Shreni Addisherla, Divya Koria]

---

## License

This project is for educational purposes as part of a capstone project.

---

## Contact

For questions or issues, please contact the project team or create an issue in the repository.

---

## Acknowledgments

- Dataset provided for educational purposes
- CRISP-DM methodology framework
- Scikit-learn and open-source ML community

