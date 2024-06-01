## End to End ML Project for Student Performance Prediction

Dataset: Student data with attributes such as gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score, math_score
Math_score is the target attribute which we need to predict.

Technology Used: Machine Learing Models, Tensorflow, Python, Flask

Flask is used for developing user interface.

ML Models trained: 
- Decision Tree
- Random Forest
- Gradient Boosting
- Linear Regressor
- XGB Regressor
- Catboost Regressor
- Adaboost Regressor

Steps:
- DataIngestion : Read the data and split it into training and test data
- DataTransformation : Handle missing data. Apply One hot encoding and normalize the values.
- ModelTraining: Initialize various models. Train them using the train data and return the best_model.
- TestPipeline: Get the new data entered by user and predict the math_score based on that using the best_model.

