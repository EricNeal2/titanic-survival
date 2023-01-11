# Titanic Survival
This is my submission to Kaggle's [Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition. Using Titanic passenger data (name, age, gender, socio-economic class, etc), I built a predictive model to answer the question “what sorts of people were more likely to survive?”

## Summary

- **Preprocessing**: Handle missing values, convert the `age`, `fare`, and `cabin` columns to categorical, extract a title from the name column, and create dummy columns for any single column.
- **Feature Engineering**: Combine the information in the `SibSp` and `Parch` columns to create a new feature, `is_alone`, which indicates whether a person was traveling alone or with any family members.
- **Feature Selection**: Using Recursive Feature Elimination, determine the optimal parameters for a random forest classifier.
- **Model Selection and Tuning**: Use a grid search to train 3 different models (Logistic Regression, K-Nearest Neighbours, Random Forest) with different hyperparameters.
- **Submission**: After determining the best model, make predictions on the holdout data and export a CSV file to submit to Kaggle.

## Results

The most accurate model was a Random Forest classifier, with the following hyperparameters:
- 'criterion': 'gini', 
- 'max_depth': 10, 
- 'max_features': 'sqrt', 
- 'min_samples_leaf': 1, 
- 'min_samples_split': 5, 
- 'n_estimators': 6

This model had a simple accuracy of 84.0% with 10-fold cross validation. When I used this model to make predictions on the holdout data, the accuracy was 76.8%. This most likely indicates that my model is overfit to the training data.
