# medical-diagnosis
The project aims to build a machine learning model for breast cancer prediction. It uses the Wisconsin Breast Cancer Diagnostic dataset to classify tumors as malignant (cancerous) or benign (non-cancerous).
This project is a breast cancer classification pipeline using supervised machine learning techniques. It involves exploratory data analysis (EDA), data preprocessing, feature scaling, and training/testing multiple classification models to predict whether a tumor is benign or malignant based on clinical features.

Here’s a step-by-step description of what your project does:
🔍 1. Data Loading and Cleaning

    Reads the dataset (data.csv) using pandas.

    Drops unnecessary columns like 'Unnamed: 32' and 'id', which don’t contribute to the prediction.

    Checks for null values and duplicates using isnull().sum() and duplicated().sum().

📊 2. Exploratory Data Analysis (EDA)

    Uses .info() and .describe() to understand the structure and summary statistics of the dataset.

    Visualizes class distribution (diagnosis) with a count plot.

    Displays a correlation heatmap to see relationships between features.

    Uses boxplots to identify potential outliers in the data across each feature.

🔄 3. Preprocessing

    Encodes the target variable diagnosis (B = 0 for benign, M = 1 for malignant) using LabelEncoder.

    Splits the dataset into features (X) and labels (y).

    Applies standardization to the features using StandardScaler.

    Splits the data into training and testing sets using train_test_split.

🤖 4. Model Building & Evaluation

Several classification models are trained and evaluated:
✅ K-Nearest Neighbors (KNN)

    Trained with default parameters.

    Predictions made and evaluated using:

        accuracy_score

        confusion_matrix

        classification_report

✅ Logistic Regression

    Trained and tested for accuracy on both training and testing datasets.

✅ Decision Tree Classifier

    Trained with default parameters and later with hyperparameter tuning using GridSearchCV (on criterion and max_depth).

    Best model used to make predictions and evaluate performance.

✅ Random Forest Classifier

    Trained and tested.

    Grid search is used for tuning n_estimators, criterion, and max_depth.

    Evaluation through score, confusion_matrix, and classification_report.

✅ Support Vector Machine (SVC)

    Trained and evaluated on training and testing data.

📈 5. Performance Metrics

    The models are evaluated using:

        Accuracy

        Confusion Matrix

        Classification Report (Precision, Recall, F1-Score)

    The confusion matrix for Random Forest is also visualized using a heatmap.

🔧 6. Optimization

    GridSearchCV is used to find the best hyperparameters for:

        Decision Tree

        Random Forest

🧠 Goal of the Project:

To predict breast cancer diagnosis based on features derived from digitized images of a breast mass, supporting early detection and medical decision-making.
✅ Summary:

    Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset.

    Target: Diagnosis (Benign or Malignant).

    Models Used: KNN, Logistic Regression, Decision Tree, Random Forest, SVM.

    Techniques: EDA, Label Encoding, Standardization, Train-Test Split, Hyperparameter Tuning.
