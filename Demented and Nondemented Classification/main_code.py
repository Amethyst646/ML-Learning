# Import currently utilized libraries
import numpy as np
import pandas as pd

from sklearn.impute import IterativeImputer

# Load dataset and create copy of dataset for future use
pdDS = pd.read_csv('/content/DementedvsNonDemented.csv')
DS = pdDS.copy()

# Check the data
DS.head(10)
DS.tail(10)
DS.info()
DS.describe()
DS.isnull().sum()
DS.nunique()
DS.duplicated().sum()

## Start Data Preprocessing
# Display data with NaN value
filtered_DS = DS[DS.isnull().any(axis=1)]
filtered_DS

# Convert variables with object data into nominal code type
DS_to_convert = ['Subject ID', 'MRI ID', 'Group', 'M/F', 'Hand']
for col in DS_to_convert:
  DS[col] = DS[col].astype('category').cat.codes
DS.info() # Confirm datatype has been succesfully converted

#Impute missing data with IterativeImputer from sklearn
DSimputer = IterativeImputer()
imputedDS = DSimputer.fit_transform(DS)
imputedDS = pd.DataFrame(imputedDS, columns=DS.columns)

#Convert continuous data on SES and MMSE into discrete data
imputedDS['SES'] = round(imputedDS['SES'])
imputedDS['MMSE'] = round(imputedDS['MMSE'])

# Check new data
imputedDS.head(5)

## Start exploratory data analysis (EDA)
# Store column names
col_name = imputedDS.columns

# Plot line/scatter
fig, axes = plt.subplots(5,3, figsize=(12,20))
for i, var in enumerate(col_name):
  row = i//3
  col = i%3
  ax = axes[row, col]
  ax.scatter(range(len(imputedDS.iloc[:,i])), imputedDS.iloc[:,i], label=col_name)
  ax.set_title(f'{var}')
