import pandas as pd
from functions import *



df = pd.read_csv('https://raw.githubusercontent.com/rasvob/VSB-FEI-Fundamentals-of-Machine-Learning-Exercises/master/datasets/titanic.csv', index_col=0)


if __name__ == '__main__':
    perform_titanic_clustering_analysis(df)