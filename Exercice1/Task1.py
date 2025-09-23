import pandas as pd

def find_undervalued_houses(file_path):
    df = pd.read_csv(file_path)
    df['Undervalued'] = (
        (df['SalePrice'] < 163000) &
        (df['OverallQual'] > 5) &
        (df['OverallCond'] > 5)
    )

    undervalued_count = df['Undervalued'].sum()

    return undervalued_count


dataset_file = 'my_data1.csv'

    
number_of_undervalued_houses = find_undervalued_houses(dataset_file)

if number_of_undervalued_houses is not None:
    print(f"Number of undervalued houses: {number_of_undervalued_houses}")
    