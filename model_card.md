# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model uses Random Forest Classifier, built by leverging scikit-learn, as part of a machine learning pipeline project. The goal was to predict whether an individual earns more than $50K per year based on their census data. I used 100 decision trees (estimators) and set a random state of 42 so the results are consistent every time the model is trained.

## Intended Use
This model was built for educational purposes as a part of a data analytics course. It is intended to practice building, evaluating, and deploying a machine learning model. It should not be used to make and real financial, hiring, or policy decisions, without much more rigorous testing and evaluation.

## Training Data
I used the Census Income dataset from teh UCI Machine Learning Repository. The dataset had about 32,000 records and 14 features. I split the data 80/20 for training and testing. Text-based, like occupation and education level, were converted to numbers using OneHotEncoding, and the salary label was converted to 0s and 1s using a LabelBinarizer.

## Evaluation Data
The model was evaluated on a 20% of the original Census Income dataset, randomly split using a random state of 42. I made sure to use the same encoder on the test data that was used to fit on the training data, so the preprocessing was consistent.

## Metrics
I evaluated the model using precision, recall, and F1 score:
-Precision: 0.7419 (when the model predicted >$50K, it was right 74% of the time)
-Recall: 0.6384 (the model caught around 64% of the people who actually earn >$50K)
-F1 Score: 0.6863 (the overall balanced score between precision and recall is 69%)

## Ethical Considerations
This dataset contains sensitive demographic features such as race, gender, and native country. Because of this, the model could reflect biases that exsisted in 1994 census data it was trained on. I ran slice analysis across all categorical features to check how the model performs for all groups. The results showed that performance does vary across the groups, which is something to be aware of if this model was ever used in a real life setting.

## Caveats and Recommendations
-The model was trained on census data from 1994, so it doesnt not reflect on today's workforce or income levels.
-Model performance is not equal across all demographic groups so the slice output (slice_output.txt) should be reviewd carefully.
-I did not perform any hyperparameter tuning, so there is room to improve the model's performance.
-This model was built as a learning project and should not be used for any real decisions.