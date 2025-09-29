import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


url = 'https://raw.githubusercontent.com/rasvob/VSB-FEI-Fundamentals-of-Machine-Learning-Exercises/master/datasets/zsu_cv1_data.csv'
df = pd.read_csv(url)

df_fam = df[df['BldgType'] == '1Fam']
df_twn = df[df['BldgType'].isin(['Twnhs', 'TwnhsE'])]

features = ['LotFrontage', 'LotArea', 'YearBuilt', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'SalePrice']

corr_fam = df_fam[features].corr() 
corr_twn = df_twn[features].corr()

def plot_correlation_matrix(corr_matrix, title):
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title(title, fontsize=16)
    plt.show()


plot_correlation_matrix(corr_fam, "Correlation Matrix for Single-Family Homes (BldgType = '1Fam')")

input("Press Enter to continue...")

plot_correlation_matrix(corr_twn, "Correlation Matrix for Townhouses (BldgType = 'Twnhs' or 'TwnhsE')")
