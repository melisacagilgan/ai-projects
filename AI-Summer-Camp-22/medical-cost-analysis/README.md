# Medical Cost Analysis and Insurance Forecast

## About Project
This project was done within Global AI Hub's AI Summer Camp'22. In this project, one of Kaggle's datasets, including personal medical costs based on age, sex, bmi, children, smoker, region, and charges, is used. First, I analyzed the dataset in both a written and visual manner. I also applied some preprocessing methods to the dataset. After that, I chose to test the following models as a regressor for the project;
- Linear Regression Model
- Ridge Regression Model
- Histogram-based Gradient Boosting Regression Tree
- Random Forest Regressor
- Bayesian Ridge Model
- AdaBoost Regressor
- Support Vector Regressor -RBF
- Support Vector Regressor -Linear

I must have chosen the best model based on the performance metrics and applied the hyperparameter optimization to that model. The highest accuracy belonged to the Histogram-based Gradient Boosting Regression Tree, so I optimized this model using GridSearchCV. The ultimate goal was to use the best model with optimized hyperparameters to forecast insurance costs using the dataset mentioned above.


## About Dataset
- Name: Medical Cost Personal Datasets
- Kaggle Link: [insurance.csv](https://www.kaggle.com/datasets/mirichoi0218/insurance?datasetId=13720)

### Content
- age: Age of primary beneficiary
- sex: Insurance contractor gender (female, male)
- bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
- children: Number of children covered by health insurance / Number of dependents
- smoker: Smoking status (yes, no)
- region: The beneficiary's residential area in the US (northeast, southeast, southwest, northwest)
- charges: Individual medical costs billed by health insurance