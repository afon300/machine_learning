import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_dwelling_type_chart(df_path: str):

    df = pd.read_csv(df_path)

    required_columns = ['YearBuilt', 'BldgType']
    if not all(col in df.columns for col in required_columns):
        print(f"The DataFrame must contain the following columns: {required_columns}")

    df['AgeGroup'] = df['YearBuilt'].apply(lambda year: 'After 2000' if year > 2000 else 'Before 2000')

    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x='BldgType', hue='AgeGroup', palette='viridis')

    plt.title('Number of Houses by Dwelling Type and Build Year', fontsize=16)
    plt.xlabel('Dwelling Type', fontsize=12)
    plt.ylabel('Number of Houses', fontsize=12)
    plt.legend(title='Build Year')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig('tab.png')
    print("Chart saved as 'dwelling_type_by_age.png'")


file_path = 'my_data.csv'

create_dwelling_type_chart(file_path)