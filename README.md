 Phase-3-project
A Data-Driven Approach to Improve Customer Retention
This project analyzes customer churn for SyriaTel, a telecommunications company, using predictive modeling techniques and data visualization. Project Overview:
This project aims to develop a classification model that will predict customer churn for SyriaTel, a telecommunications company. I have chosen to follow the CRISP-DM method to complete this project. It will involve six stages: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment. The project purposes to provide insights into the patterns and factors influencing customer churn, and also develop a predictive model to assist in reducing customer attritione goal is to understand patterns in customer behavior and develop a model to predict churn.
Business Understanding:
SyriaTel is the major stakeholder for this project. They are interested in reducing customer churn. By helping them predict customer churn, they can take proactive measures to ensure maximum customer retentions and profit maximization. The project majorly focuses on identifying patterns that facilitate to customer churn and providing recommendations on how to mitigate this.
1.Problem Definition
Objective: Develop a predictive model to determine whether a customer will churn (binary classification: Yes/No) based on customer usage patterns, interaction with the company, and plan features.

Outcome: Provide actionable insights to SyriaTel to reduce customer churn by identifying high-risk customers and enabling targeted retention strategies.

Metric for Success: Evaluate the model's performance using metrics such as:

Accuracy: Measures overall correctness but may not address class imbalance. Precision: Useful when minimizing false positives (e.g., targeting non-churners for retention campaigns is costly). Recall: Important to identify as many churners as possible (minimizing false negatives). F1-Score: Balances Precision and Recall, suitable for imbalanced datasets. AUC-ROC: Evaluates the tradeoff between true positive and false positive rates across thresholds.
Data Understanding
Data understanding lets us explore and analyze our churn data to gain insights into its structure, content, and relationships. It involves looking at the types of data and what the columns entail, identifying patterns, checking for missing values, and understanding the distribution of variables. The goal is to familiarize ourselves with the data before any analysis or modeling, ensuring that we can make informed decisions and address any issues, such as imbalances or outliers, that might affect the results.
Data Preparation
-Identifying and removing duplicate rows. -Handling missing/NAN values to ensure data consistency. -Eliminating irrelevant columns that do not contribute meaningfully to the analysis.
checking for relation against categorical columns
checking for relation against numerical columns
data understanding
The dataset has 3333 rows and 21 columns and has no null values or duplicates. Therefore we do not need to impute any missing values or drop any duplicated values in this case. Among the 21 columns five of them are categorical in nature; 'state', 'phone number', 'international plan', 'voice mail plan','churn'. Churn which is our target variable in the data set is of boolean data type. Thus, we will make it binary later when building our models.

Some of the columns based on domain knowledge are not actually good predictors and thus dropping them before fitiing into our models will be good. For example, the phone number variable has nothing to do with customer chruning the company.

Most values in the dataset are numerical in nature. The summary statistics provides a brief overview of the dataset and the range of values observed in each numerical column.
Exploratory Data Analysis
Exploratory Data Analysis (EDA) on our dataset will involve examining the churn data to understand the underlying patterns, relationships, and characteristics of the features before building any predictive models. This includes analyzing the distribution of numerical variables (e.g., total day minutes, total night calls), understanding categorical features (e.g., international plan, voice mail plan), identifying potential outliers or anomalies, checking for missing data, and exploring correlations between features. EDA also involves visualizing the data using tools like histograms, boxplots, and correlation heatmaps to uncover trends or patterns that could influence customer churn prediction, ultimately helping to make informed decisions about data preparation and modeling strategies.

We then generate boxplots to detect outliers in numerical features, helping visualize data distribution and identify extreme values that could impact analysis or modeling. By highlighting outliers, it guides data cleaning steps such as removing, transforming, or imputing extreme values and informs decisions about scaling or normalizing features to ensure consistency.

The target variable is churn. It is binary variable, hence we'll be solving a classification problem. Let's take a look at distribution of churn.
The analysis of the churn variable reveals that 85.51% of customers do not churn, while 14.49% of customers churn from the company.

This indicates an imbalance in the distribution of the binary classes. To address this issue and prevent the model from making false predictions, we will need to apply class imbalance treatment techniques.
Churn by International Plan
Based on the bargraph above, it is evident that customers without an international plan have a higher percentage in both the 'False' and 'True' categories compared to customers with an international plan. This suggests that having an international plan may be associated with a lower likelihood of churn.
Churn by Voice Mail Plan
From the graph above, it can be observed that the majority of customers who do not have a voice mail plan are in the 'False' category, while a smaller proportion is in the 'True' category. In addition, customers with a voice mail plan have a higher count in the 'False' category compared to the 'True' category. This may suggest that having a voice mail plan may have some influence on reducing churn,
Distribution of numeric variables
It helps us understand the range, variability, and shape of the variables. Analyzing the distribution can aid in identifying outliers, skewness, or patterns in the data.
The numerical variables exhibit diverse distributions and ranges, indicating variations in customer behavior and call patterns. While some variables follow approximately normal distributions, others display skewed distributions. This suggests that the variables may require different handling approaches based on their distributions for further analysis and modeling.
checking for outliers
Correlation Matrix
The correlation matrix reveals the relationships between variables, indicating how they are associated with each other.
From the above correlation matrix, we can observe that most of the variables are not strongly correlated. However, there are some variables that exhibit a perfect correlation. This makes sense since some variables are directly correlated.
Dropped columns: ['total day charge', 'total eve charge', 'total night charge', 'total intl charge']
we identify highly correlated columns with a correlation greater than 0.9, which are considered highly redundant, and drop them from the dataset. The goal is to reduce multicollinearity and simplify the dataset by removing highly correlated features that may not provide additional useful information for modeling.
Addressing Multicollinearity
The original dataframe has 14 columns.
The reduced dataframe has 14 columns.
No high multicollinearity was detected — based on your threshold (often a correlation > 0.9 or 0.95).

This means none of the features were strongly correlated enough with each other to justify dropping one.
label encoding
Transforming churn values into 0s and 1s so the data is compatible with the models enabling them to perform calculations and predictions. Many algorithms, especially classification models (e.g., logistic regression, decision trees, and random forests), require numeric inputs for target variables.
one hot encoding
we convert the categorical values "yes" and "no" in the 'international plan' and 'voice mail plan' columns into numerical representations (1 for "yes" and 0 for "no"). This transformation makes the data suitable for machine learning algorithms, which typically require numerical inputs. By applying this mapping, the code prepares these categorical features for modeling while maintaining the information they represent.
Modelling
In the modeling step, we will train and evaluate different machine learning models on our dataset to make predictions for the target variable. This involves selecting appropriate algorithms, tuning their parameters, and assessing their performance using various evaluation metrics. The goal is to find the model that best captures the patterns and relationships in the data and provides accurate predictions.
Model 1. Logistic Regression
Training set class distribution:
 0.0    1883
1.0     214
Name: churn, dtype: int64 

Test set class distribution:
 0.0    610
1.0     90
Name: churn, dtype: int64
This gives an overview of how the target variable (churn) is distributed across both training and test sets, showing how balanced or imbalanced the data is for each class (e.g., the number of churn vs. non-churn instances). It can also be a guidance as to whether further techniques like class balancing are needed.
**************** LOGISTIC REGRESSION CLASSIFIER MODEL RESULTS ****************
              precision    recall  f1-score   support

    No Churn       0.88      0.99      0.93       610
       Churn       0.53      0.09      0.15        90

    accuracy                           0.87       700
   macro avg       0.71      0.54      0.54       700
weighted avg       0.84      0.87      0.83       700
Class: No Churn

1.Precision (0.88): 88% of the customers predicted as No Churn were actually No Churn.

2.Recall (0.99): The model correctly identified 99% of actual No Churn customers.

3.F1-score (0.93): This is a harmonic mean of precision and recall — and shows very strong performance on this class.

Class: Churn

1.Precision (0.53): Only 53% of the predicted churns were correct.

2.Recall (0.09): Only 9% of actual churns were correctly identified. This is very low.

3.F1-score (0.15): The model is doing poorly on this class — it's barely catching churners.

Overall Accuracy: 87%

Sounds good, but very misleading due to class imbalance.

The model is biased toward predicting "No Churn" because the majority of customers don’t churn.

The confusion matrix indicates the performance of the model as follows:

1.True Negatives (No Churn correctly predicted): 603 instances were correctly classified as "No Churn."

2.False Positives (Predicted Churn but was No Churn): 7 instances were incorrectly classified as "Churn" when they were actually "No Churn."

3.False Negatives (Predicted No Churn but was Churn): 82 instances were incorrectly classified as "No Churn" when they were actually "Churn."

4.True Positives (Churn correctly predicted): 8 instances were correctly classified as "Churn."

Overall, the model performs well for identifying "No Churn" instances but struggles significantly with identifying "Churn," as evidenced by the high number of false negatives. This suggests a potential imbalance in the dataset or room for improvement in model sensitivity towards the "Churn" class.
**************** LOGISTIC REGRESSION CLASSIFIER MODEL RESULTS ****************
Accuracy: 0.87286
Precision: 0.53333
Recall: 0.08889
F1 Score: 0.15238
The Logistics Regressionmodel performance metrics are as follows:

Accuracy (0.87286): The model accurately predicted 87% of all instances. Precision (0.5333): Of the cases predicted as "Churn," 53% were correct. Recall (0.08889): The model successfully identified only 8% of the actual "Churn" cases. F1 Score (0.15238): The low F1 score reflects poor overall performance in detecting "Churn," balancing both precision and recall.

These metrics are particularly useful for imbalanced datasets, as Accuracy alone may not reflect the model's ability to correctly identify the minority class ("Churn"). In this case, the metrics will highlight that while the model performs well in predicting "No Churn," it has lower Recall and F1 Score for "Churn," indicating room for improvement in recognizing this minority class.
Applying SMOTE Technique to Resolve Unbalanced 'churn' Feature
precision    recall  f1-score   support

    No Churn       0.97      0.79      0.88       624
       Churn       0.33      0.83      0.47        76

    accuracy                           0.80       700
   macro avg       0.65      0.81      0.67       700
weighted avg       0.90      0.80      0.83       700

Class: No Churn

1.Precision (0.97):

Out of all customers predicted as not churning, 97% actually didn’t churn. Excellent precision.

2.Recall (0.79):

Out of all real non-churners, 79% were correctly identified. A few were wrongly predicted as churners, but still strong.

3.F1-score (0.88):

A solid balance between precision and recall.

Class: Churn (Minority Class)

1.Precision (0.33):

Only 33% of predicted churners were actually churners. The model is still making some false positives.

2.Recall (0.83):

This is excellent! The model now detects 83% of real churners, compared to only 9% before SMOTE. Huge improvement in catching churn!

3.F1-score (0.47):

Still modest, but better than before.

Tells you there's room to improve precision, but recall is strong.
Confusion Matrix:
 [[496 128]
 [ 13  63]]
The confusion matrix indicates the performance of the model as follows:

1.True Negatives (No Churn correctly predicted): 496 instances were correctly classified as "No Churn."

2.False Positives (Predicted Churn but was No Churn): 128 instances were incorrectly classified as "Churn" when they were actually "No Churn."

3.False Negatives (Predicted No Churn but was Churn): 13 instances were incorrectly classified as "No Churn" when they were actually "Churn."

4.True Positives (Churn correctly predicted): 63 instances were correctly classified as "Churn."

**************** LOGISTIC REGRESSION CLASSIFIER MODEL RESULTS ****************
Accuracy: 0.79857
Precision: 0.32984
Recall: 0.82895
F1 Score: 0.47191
The Logistics Regression model performance metrics are as follows:

Accuracy (0.79857): The model accurately predicted 79.86% of all instances. Precision (0.32984): Of the cases predicted as "Churn," 32.98% were correct. Recall (0.82895): The model successfully identified only 82.90% of the actual "Churn" cases. F1 Score (0.47191):

Accuracy 80% of all predictions are correct — but this can be misleading with imbalanced classes.

Precision 33% of customers predicted as churners actually churned (many false positives).

Recall 83% of actual churners were correctly identified — very good recall.

F1 Score 0.47 — a balance between precision and recall; still moderate overall performance.

Model 2 decision tree
This classifier is predicting whether a customer will Churn or Not Churn.
Best Parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 10}
              precision    recall  f1-score   support

    No Churn       0.97      0.88      0.92       624
       Churn       0.43      0.75      0.54        76

    accuracy                           0.86       700
   macro avg       0.70      0.81      0.73       700
weighted avg       0.91      0.86      0.88       700

Precision: Of the predicted class instances, how many were correct.

For Churn, 43% of the predicted churns were actual churns.

Recall: Of the actual class instances, how many were captured.

For Churn, the model caught 75% of actual churn cases.

F1-score: Harmonic mean of precision and recall (balance between them).

key insights
The model performs very well on the majority class (No Churn).

Performance on minority class (Churn) is much lower, especially in precision (many false positives).

Recall for Churn is decent (0.75), meaning it’s catching most of the true churn cases—but at the cost of misclassifying some No Churn customers.

This is a common trade-off in imbalanced classification problems.
**************** DECISION TREE CLASSIFIER MODEL RESULTS ****************
Accuracy: 0.87
Confusion Matrix:
 [[554  70]
 [ 21  55]]
Classification Report:
               precision    recall  f1-score   support

    No Churn       0.96      0.89      0.92       624
       Churn       0.44      0.72      0.55        76

    accuracy                           0.87       700
   macro avg       0.70      0.81      0.74       700
weighted avg       0.91      0.87      0.88       700

No Churn:

Precision 0.96: 96% of predicted No Churn were correct.

Recall 0.89: 89% of actual No Churn customers were correctly identified.

F1-score 0.92: Strong overall performance.

Churn:

Precision 0.44: Only 44% of predicted Churn were actually Churn. This means many false positives.

Recall 0.72: The model caught 72% of real churn cases.

F1-score 0.55: Moderate performance—better recall than precision.

Key Takeaways

Your model performs very well for the majority class (No Churn).

Churn prediction still needs work:

Recall is decent (72%) → Good at finding churners.

Precision is low (44%) → Predicts too many false churners, which may be costly if actions are taken based on this.

MODEL 3 A Random Forest Model
**************** RANDOM FOREST CLASSIFIER RESULTS ****************
Accuracy: 0.94143
Precision: 0.72152
Recall: 0.75000
F1 Score: 0.73548
The Random Forest model which has SMOTE applied to it clearly outperforms the Logistic Regression baseline in all metrics. While Logistic Regression shows acceptable accuracy, its poor recall and F1 score highlight its inability to effectively detect "Churn." In contrast, Random Forest demonstrates strong performance across all metrics, making it a much better choice for this problem, especially if identifying "Churn" is critical.

Best Hyperparameters: {'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
**************** TUNED RANDOM FOREST CLASSIFIER RESULTS ****************
Accuracy: 0.94429
Precision: 0.76056
Recall: 0.71053
F1 Score: 0.73469
The tuned Random Forest model achieves excellent performance across all metrics, significantly improving the detection of "Churn" compared to earlier models like Logistic Regression. It effectively balances precision and recall, making it reliable for applications where identifying churners accurately is critical for business strategy. The chosen hyperparameters likely enhanced the model's ability to generalize and capture the complexities of the data.

Hyperparameter tuning marginally improved overall performance, with a higher accuracy, precision, and F1 score compared to the baseline. While recall slightly decreased, the improvement in precision ensures that the tuned model is more reliable and consistent in its predictions. This makes the tuned Random Forest classifier a more robust choice, especially in scenarios prioritizing reduced false positives without sacrificing much recall.

Feature Importance according to the random forest model
                 Feature  Importance
4        total day minutes    0.258197
1       international plan    0.122312
6        total eve minutes    0.102947
11        total intl calls    0.096494
8      total night minutes    0.070882
10      total intl minutes    0.065523
0           account length    0.046948
5          total day calls    0.045414
9        total night calls    0.044830
12  customer service calls    0.044114
7          total eve calls    0.040950
3    number vmail messages    0.038998
2          voice mail plan    0.022390

Top Contributors:
total day minutes (0.265): This feature has the highest importance, meaning the total minutes a customer spends on daytime calls is the most critical factor in predicting churn. international plan (0.150): Whether a customer has subscribed to an international plan is the second most influential factor, reflecting its impact on churn decisions. total intl calls (0.101): The total number of international calls made is another significant factor, showing its relevance in customer churn behavior.

Less Significant Features:
Call and account-related features like total day calls (0.048), account length (0.047), and total eve calls (0.043) have lower importance, suggesting they are less predictive of churn compared to the top features. customer service calls (0.035): While low, this feature still has some influence, as frequent interactions with customer service might be a signal of dissatisfaction. voice mail plan (0.035): This feature has minimal impact, indicating it is not a major factor in predicting churn.

Summary:
The model emphasizes call usage patterns (minutes and international calls) and subscription plans (international plan) as the primary predictors of churn. Features like account length, voice mail plan, and customer service calls have relatively less influence. These insights could guide strategies for churn reduction by focusing on optimizing services related to the most critical features.


*************** MODEL COMPARISON RESULTS ***************
Training Data:
              classifiers      auc  accuracy
0      LogisticRegression  0.86147  0.904149
1  RandomForestClassifier  1.00000  1.000000
2  DecisionTreeClassifier  1.00000  1.000000

Best Model on Training Data: DecisionTreeClassifier (AUC: 1.000, Accuracy: 1.000)

Test Data:
              classifiers       auc  accuracy
0      LogisticRegression  0.864604  0.898571
1  RandomForestClassifier  0.917858  0.962857
2  DecisionTreeClassifier  0.834599  0.921429

Best Model on Test Data: RandomForestClassifier (AUC: 0.918, Accuracy: 0.963)

Of the three models (Logistic Regression, Random Forest, and Decision Tree) based on their AUC and accuracy scores for both training and test data we can conclude as follows;

Training Data Results: Random Forest and Decision Tree models have perfect accuracy and AUC scores of 1.000, suggesting they fit the training data perfectly. However, this could indicate overfitting, as these models may have memorized the training data rather than generalizing well. Test Data Results: When evaluated on the test data, the Random Forest classifier stands out with the highest AUC (0.9179) and accuracy (96.28%). It outperforms the other two models, indicating better generalization and performance on unseen data. The Decision Tree model has a relatively high accuracy but a lower AUC, suggesting it may not handle the complexity of the data as well as Random Forest. Logistic Regression also has a lower AUC and accuracy compared to Random Forest. Conclusion: The Random Forest classifier is the best model to use, as it achieves the highest accuracy and AUC on both training and test data, with the best generalization capability to unseen data.