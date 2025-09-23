import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


########## TASK 2 ##########

with open('data.csv', 'r') as f:
    data = f.read()

# Load the dataset stock in in the file data.csv

def show_graph(dx, dy, ddata):
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(x=dx, y=dy, data=ddata)
    plt.title('Relationship between SalePrice and OverallQual')
    
    plt.xlabel('Overall Quality')
    plt.ylabel('Sale Price ($)')
    
    plt.show()

df = pd.read_csv(data)

show_graph('OverallQual', 'SalePrice', df)

x = input("please press any key to continue ...")

show_graph('OverallCond', 'SalePrice', df)