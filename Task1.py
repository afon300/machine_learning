import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

########## TASK 1 ##########

# open the file data.csv and read its content
with open('data.csv', 'r') as f:
    data = f.read()

# Load the data into a Pandas DataFrame give in the jupyter notebook
df = pd.read_csv(data)

# Calculate Q1, Q3, and IQR for SalePrice
Q1_price = df['SalePrice'].quantile(0.25)
Q3_price = df['SalePrice'].quantile(0.75)
IQR_price = Q3_price - Q1_price

# same for GrLivArea
Q1_area = df['GrLivArea'].quantile(0.25)
Q3_area = df['GrLivArea'].quantile(0.75)
IQR_area = Q3_area - Q1_area

lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price
lower_bound_area = Q1_area - 1.5 * IQR_area
upper_bound_area = Q3_area + 1.5 * IQR_area

# Mark outliers
df['is_outlier'] = ((df['SalePrice'] < lower_bound_price) | (df['SalePrice'] > upper_bound_price)) | \
                   ((df['GrLivArea'] < lower_bound_area) | (df['GrLivArea'] > upper_bound_area))

plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df, hue='is_outlier', palette={True: 'red', False: 'blue'}, alpha=0.6)
plt.title('Outlier Detection for SalePrice and GrLivArea')
plt.xlabel('Ground Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
plt.show()

# Print the number of outliers detected
print(f"Number of outliers detected: {df['is_outlier'].sum()}")