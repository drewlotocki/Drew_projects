# Necessary Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64

# Makes the dataframe for the sankey diagram
data = pd.read_csv('heart_attack_prediction_dataset.csv') # Opens csv
data = data.drop(['Patient ID', 'Blood Pressure','Heart Rate', 'Cholesterol', 'Hemisphere', ], axis = 1)
data['Income'] = data['Income'].round(-5) # Bins incomes
data['Diabetes'] = data['Diabetes'].replace({0: 'Not Diabetic', 1: 'Diabetic'}) # Changing all boolean labels to readable codes
data['Smoking'] = data['Smoking'].replace({0: 'Not a Smoker', 1: 'Smoker'})
data['Obesity'] = data['Obesity'].replace({0: 'Not Obese', 1: 'Obese'})
data['Alcohol Consumption'] = data['Alcohol Consumption'].replace({0: 'Low Alcohol Consumption', 1: 'High Alcohol Consumption'})
data['Family History'] = data['Family History'].replace({0: 'No History', 1: 'Has History'})
data['Previous Heart Problems'] = data['Previous Heart Problems'].replace({0: 'No Problems', 1: 'History of Problems'})
data['Diet'] = data['Diet'].replace({'Healthy': 'Healthy Diet', 'Average': 'Average Diet', 'Unhealthy': 'Unhealthy Diet'})
data['Medication Use'] = data['Medication Use'].replace({0: 'No Medication Use', 1: 'Medication Use'})
data = data.drop('Exercise Hours Per Week', axis=1) # Getting rid of unwanted columns
data = data.drop('Sedentary Hours Per Day', axis=1)
data['Age'] = data['Age'].round(-1) # Rounds age to 10


# Makes country counts data frame for map
columns = list(data.columns)

country_data = data.loc[:, ['Country']]
country_data = country_data.groupby(['Country']).size().reset_index(name='Count')


# Data for regression plot and png
df_raw = pd.read_csv("heart_attack_prediction_dataset.csv")
x_feat_list = ['Age', 'Sex', 'Cholesterol',
               'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
               'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
               'Medication Use', 'Stress Level',
               'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
               'Physical Activity Days Per Week', 'Sleep Hours Per Day']  # no blood pressure column here

# Convert DataFrame columns  (Strings to Integers)
# Mapping for 'Sex' column
sex_mapping = {'Male': 0, 'Female': 1}
df_raw['Sex'] = df_raw['Sex'].map(sex_mapping)

# Mapping for 'Diet' column
diet_mapping = {'Healthy': 3, 'Average': 2, 'Unhealthy': 1}
df_raw['Diet'] = df_raw['Diet'].map(diet_mapping)

# Calculate RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

x = df_raw.loc[:, x_feat_list].values
# prices (target)
y = df_raw['Previous Heart Problems']

rfr = RandomForestRegressor()
rfr.fit(x, y)


def plot_feat_import(feat_list, feat_import, sort=True, limit=None):
    if sort:
        idx = np.argsort(feat_import).astype(int)
        feat_list = [feat_list[_idx] for _idx in idx]
        feat_import = feat_import[idx]

    if limit is not None:
        feat_list = feat_list[:limit]
        feat_import = feat_import[:limit]

    plt.figure(figsize=(8, 6))
    plt.barh(feat_list, feat_import)
    plt.xlabel('Feature importance (Mean MSE reduction)')
    plt.title('Feature Importance')
    plt.tight_layout()

    plt.savefig('feature_importance.png')
    return plt


# Generate the RFR plot and convert to png
plt_obj = plot_feat_import(x_feat_list, rfr.feature_importances_)


with open('feature_importance.png', 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()