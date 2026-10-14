# Heart Attack Risk Prediction

## Project Overview

This project explores heart-disease data using both **supervised** and **unsupervised machine learning**. Supervised models learn to predict the dataset's `output` label from patient features, while unsupervised methods explore patterns and possible groups without using that label as an input.

This is an educational machine-learning project. Its predictions are not a medical diagnosis and should not be used to make healthcare decisions.

## Objective

- Train and compare supervised classification models for predicting the dataset's `output` label.
- Explore the structure of the data using PCA, K-Means, and DBSCAN.
- Evaluate model performance and visualise important results.

## Dataset

The project uses `heart.csv`, which contains patient attributes such as:

- Age and sex
- Chest-pain type
- Resting blood pressure and cholesterol
- Fasting blood sugar and resting ECG
- Maximum heart rate and exercise-induced angina
- ST depression, slope, number of major vessels, and thalassemia-related code

The dataset has **303 original records**. After removing one exact duplicate, **302 records** remain, with 13 input features and the target column `output`. The exact source of the dataset should be documented from the original download or dataset page.

## Machine Learning Workflows

### 1. Unsupervised Learning

Notebook: `01_Unsupervised_Learning.ipynb`

The target column `output` is excluded from the clustering input. The workflow encodes categorical features, scales the features, and uses Principal Component Analysis (PCA) to visualise the data in two dimensions. K-Means and DBSCAN are then used to explore groups and possible outliers. The known labels are compared with the discovered clusters only after clustering.

**Recorded unsupervised results:**

| Method / metric | Result |
|---|---:|
| PCA — PC1 variance explained | 24.47% |
| PCA — PC2 variance explained | 13.65% |
| PCA — first two components combined | 38.12% |
| K-Means clusters | 2 |
| K-Means cluster sizes | 167 and 135 |
| K-Means silhouette score | 0.1648 |
| K-Means Adjusted Rand Index (ARI) | 0.2508 |
| DBSCAN main clusters | 1 |
| DBSCAN noise records | 56 |
| DBSCAN main cluster size | 246 |
| DBSCAN Adjusted Rand Index (ARI) | 0.0554 |
| DBSCAN silhouette score | Not defined for the single non-noise cluster |

The clustering scores show that the discovered groups have limited agreement with the known `output` labels. These methods are exploratory and do not diagnose heart disease.

### 2. Supervised Learning

Notebook: `02_supervised_learning.ipynb`

The supervised workflow checks and prepares the dataset, scales continuous features, one-hot encodes coded categorical features, and uses a stratified training, validation, and test split. It compares Logistic Regression, Decision Tree, Random Forest, SVM variants, MLP, and Bagging. Random Forest and RBF-kernel SVM are tuned using cross-validation. The selected model is evaluated on a held-out test set.

**Final recorded test results for the tuned RBF-SVM:**

| Metric | Result |
|---|---:|
| Accuracy | 83.61% |
| Precision | 81.08% |
| Recall | 90.91% |
| F1-score | 85.71% |
| ROC-AUC | 93.51% |
| Test set size | 61 records |

These scores are estimates from a small test set and do not establish clinical performance.

### 3. Final Demonstration

Notebook: `03_final_demo.ipynb`

The final demonstration loads the saved model pipeline and uses it to make a prediction for a sample patient record. In the recorded example, the predicted label is `1`, with a model probability estimate of approximately **83.4%**. This is a model estimate for the dataset label, not a clinically validated heart-attack probability.

## Project Structure

```text
heart-attack-risk-prediction/
├── data/
│   └── heart.csv
├── notebooks/
│   ├── 01_Unsupervised_Learning.ipynb
│   ├── 02_supervised_learning.ipynb
│   └── 03_final_demo.ipynb
├── results/
│   ├── correlation_matrix.png
│   ├── dbscan_plot.png
│   ├── kmeans_plot.png
│   └── pca_plot.png
├── heart_attack_prediction_model.pkl
├── requirements.txt
└── README.md
```

The tree above describes the intended organisation. Adjust the paths and filenames if the actual repository structure differs.

## Technologies Used

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook / Google Colab

## How to Run

1. Clone or download the repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Place `heart.csv` in the location expected by the notebooks.
4. Open the notebooks in Jupyter or Google Colab and run the cells in order.
5. For the final demonstration, ensure the saved model pipeline is available at the path expected by `03_final_demo.ipynb`.

## Result Visualisations

The `results/` folder contains the following unsupervised-learning visualisations:

- `correlation_matrix.png` — shows correlations between dataset features.
- `pca_plot.png` — displays the data projected onto the first two principal components.
- `kmeans_plot.png` — shows the K-Means cluster assignments.
- `dbscan_plot.png` — shows the DBSCAN cluster and noise assignments.

The supervised workflow also evaluates classification performance using metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC. Add the supervised plots and model-comparison files to `results/` if they are present in the repository.

## Limitations

- The dataset is small and may not represent all patient populations.
- PCA's two-dimensional view retains only 38.12% of the processed data's variance.
- K-Means and DBSCAN results depend on their parameter settings, and the recorded clusters show limited agreement with the known labels.
- The models have not been clinically validated and must not be used for diagnosis or treatment decisions.

## Team Members

- Srijan Akshit — PES1UG24AM284
- D R Rushmitha — PES1UG24AM910
