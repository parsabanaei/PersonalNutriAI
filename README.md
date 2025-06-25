# PersonalNutri AI - Health Risk Assessment System

**Author:** Parsa Banaei  
**Course:** CPSC 483 - Machine Learning  
**Project:** Integrated Health Risk Assessment and Personalized Nutrition Recommendation System

## Project Overview

PersonalNutri AI is a comprehensive machine learning system that predicts health risks for obesity, diabetes, and cardiovascular disease using basic personal information. The system achieves 98.4% accuracy on obesity prediction and 87.6% AUC on heart disease prediction using real medical datasets.

## Code Layout

### Main Files

**`PersonalNutri_AI.ipynb`** - Complete Google Colab notebook containing:
- **Cells 1-2:** Environment setup and library imports
- **Cells 3-5:** Data loading functions for NHANES (.xpt) and Framingham (.csv) files
- **Cell 6:** Dataset loading and initial exploration
- **Cell 7:** Data preprocessing and feature engineering
- **Cell 8:** Feature preparation for machine learning
- **Cell 9:** Multi-algorithm model training and comparison
- **Cell 10:** Diabetes risk calculation (rule-based)
- **Cell 11:** Heart disease model training
- **Cell 12:** Integrated health risk assessment function
- **Cell 13:** Visualization and display functions
- **Cell 14:** Comprehensive system testing with diverse profiles
- **Cell 15:** Interactive user input function
- **Cell 16:** Performance summary and results
- **Cell 17:** Model export for web application

### Exported Models Directory

**`exported_models/`** - Contains trained models ready for deployment:
- **`obesity_risk_model.pkl`** - Best performing obesity classifier (Logistic Regression)
- **`obesity_scaler.pkl`** - StandardScaler for obesity model features
- **`obesity_imputer.pkl`** - SimpleImputer for handling missing values
- **`activity_encoder.pkl`** - LabelEncoder for activity level categories
- **`heart_disease_model.pkl`** - Random Forest model for cardiovascular risk
- **`heart_disease_scaler.pkl`** - StandardScaler for heart disease features
- **`model_metadata.json`** - Model performance metrics and configuration

## Required Datasets

**Note:** Datasets are NOT included in submission as they are publicly available. Download from:

1. **NHANES Demographics (DEMO_J.xpt):**
   - Source: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&CycleBeginYear=2017
   - Size: ~12MB
   - Place in: `/content/sample_data/datasets/DEMO_J.xpt`

2. **NHANES Body Measurements (BMX_J.xpt):**
   - Source: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2017
   - Size: ~8MB
   - Place in: `/content/sample_data/datasets/BMX_J.xpt`

3. **Framingham Heart Study (framingham.csv):**
   - Source: https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset?resource=download
   - Size: ~400KB
   - Place in: `/content/sample_data/datasets/framingham.csv`

## How to Run

### Option 1: Google Colab (Recommended)
1. Upload `PersonalNutri_AI.ipynb` to Google Colab
2. Download and upload the three required datasets to the specified paths
3. Run all cells in sequence
4. Models will be automatically exported to downloadable files

### Option 2: Local Environment
1. Install required packages: `pip install pandas numpy scikit-learn plotly seaborn xport matplotlib`
2. Ensure Python 3.9+ is installed
3. Download datasets and update file paths in notebook
4. Run notebook in Jupyter environment

## System Requirements

- **Python:** 3.9+
- **Memory:** 4GB+ RAM recommended for dataset processing
- **Storage:** 2GB for datasets and models
- **Libraries:** See requirements in Cell 1 of notebook

## Key Features

- **Multi-Risk Assessment:** Obesity, diabetes, and heart disease prediction
- **Real Medical Data:** Uses NHANES and Framingham research datasets
- **High Performance:** 98.4% accuracy, 99.88% AUC scores
- **Interactive Visualization:** Plotly gauge charts with risk explanations
- **Clinical Validation:** Results align with medical guidelines
- **Export Ready:** Trained models ready for web application integration

## Performance Results

| Model | Accuracy | AUC | Status |
|-------|----------|-----|--------|
| Obesity (Logistic Regression) | 98.4% | 99.88% | ✅ Exceeds target |
| Heart Disease (Random Forest) | 93.8% | 87.6% | ✅ Exceeds target |
| Diabetes (Rule-based) | N/A | Clinical validation | ✅ Medically sound |

## Usage Example

```python
# After running all cells, assess health risks:
results = assess_health_risks(
    age=45, 
    gender="Male", 
    height_inches=70, 
    weight_pounds=220,
    activity_level="Sedentary", 
    family_history=False
)

# Display results
display_risk_results(results)

# Create visualization
fig = create_risk_visualization(results)
fig.show()
```

## Technical Architecture

- **Data Processing:** pandas, numpy for NHANES .xpt file handling
- **Machine Learning:** scikit-learn with multiple algorithm comparison
- **Visualization:** plotly for interactive health dashboards
- **Validation:** External validation on Framingham dataset
- **Integration:** Unified prediction pipeline with error handling

## Academic Contributions

1. **Novel Integration:** First unified system for multiple health risk assessment
2. **Real Data Application:** Authentic medical datasets vs. synthetic data
3. **Clinical Validity:** Combines ML with established medical guidelines
4. **Explainable AI:** Provides interpretable health risk explanations
5. **Performance Excellence:** Exceeds typical academic project standards

## Contact

**Parsa Banaei**  
CPSC 483 - Machine Learning Project  
For questions about implementation or methodology, refer to the comprehensive documentation in the notebook cells.

## License

This project is for academic purposes. Datasets are publicly available under their respective licenses. Code is available for educational use and extension.