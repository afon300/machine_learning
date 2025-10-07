import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster
import sklearn.metrics
import sklearn.preprocessing

def perform_titanic_clustering_analysis(df):
 
    # Selection
    features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df_clustering = df[features].copy()
    df_clustering.dropna(inplace=True)
    df_clustering['Sex'] = df_clustering['Sex'].astype(str)
    df_clustering['Embarked'] = df_clustering['Embarked'].astype(str)

    # Standardization
    col_names_to_scale = ['Fare', 'Age', 'SibSp', 'Parch']
    numerical_features = df_clustering[col_names_to_scale]
    scaler = sklearn.preprocessing.StandardScaler()
    X_scaled_part = pd.DataFrame(scaler.fit_transform(numerical_features), columns=col_names_to_scale, index=df_clustering.index)

    sex_encoded = sklearn.preprocessing.OrdinalEncoder().fit_transform(df_clustering[['Sex']])
    sex_encoded_df = pd.DataFrame(sex_encoded, columns=['Sex'], index=df_clustering.index)

    embarked_encoded = pd.get_dummies(df_clustering['Embarked'], prefix='Embarked')


    X_for_clustering = pd.concat([X_scaled_part, sex_encoded_df, embarked_encoded], axis=1).values


    clustering_scores = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_for_clustering)
        clustering_scores.append({
            'k': k,
            'sse': kmeans.inertia_,
            'silhouette': sklearn.metrics.silhouette_score(X_for_clustering, kmeans.labels_)
        })
    df_scores = pd.DataFrame(clustering_scores).set_index('k')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    

    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('SSE (Inertia)', color=color)
    ax1.plot(df_scores.index, df_scores['sse'], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(df_scores.index, df_scores['silhouette'], color=color, marker='x')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Elbow Method for Optimal k (Standardized Data)')
    plt.savefig('elbow_method.png')
    plt.close()
    print("Saved elbow_method.png")

    ideal_k = 4
    print(f"\nfinal cluster k={ideal_k}...")
    kmeans_final = sklearn.cluster.KMeans(n_clusters=ideal_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X_for_clustering)


    df_analysis = df_clustering.copy()
    df_analysis['cluster_id'] = clusters
    
    print("Generating images...")
    
    print("Plot 1: Fare distribution")
    plt.figure(figsize=(12, 7)); sns.boxplot(data=df_analysis, x='cluster_id', y='Fare'); plt.title('Fare Distribution by Cluster', fontsize=16); plt.yscale('log'); plt.xlabel('Cluster ID', fontsize=12); plt.ylabel('Fare (Log Scale)', fontsize=12); plt.savefig('fare_by_cluster.png'); plt.close()

    print("Plot 2: Age distribution")
    plt.figure(figsize=(12, 7)); sns.boxplot(data=df_analysis, x='cluster_id', y='Age'); plt.title('Age Distribution by Cluster', fontsize=16); plt.xlabel('Cluster ID', fontsize=12); plt.ylabel('Age', fontsize=12); plt.savefig('age_by_cluster.png'); plt.close()

    print("Plot 3: Survival rate")
    plt.figure(figsize=(12, 7)); sns.countplot(data=df_analysis, x='cluster_id', hue='Survived', palette='viridis'); plt.title('Survival Count by Cluster (0=No, 1=Yes)', fontsize=16); plt.xlabel('Cluster ID', fontsize=12); plt.ylabel('Count', fontsize=12); plt.legend(title='Survived'); plt.savefig('survival_by_cluster.png'); plt.close()

    print("Plot 4: Passenger class distribution")
    plt.figure(figsize=(12, 7)); sns.countplot(data=df_analysis, x='cluster_id', hue='Pclass', palette='plasma'); plt.title('Passenger Class Distribution by Cluster', fontsize=16); plt.xlabel('Cluster ID', fontsize=12); plt.ylabel('Count', fontsize=12); plt.legend(title='Pclass'); plt.savefig('pclass_by_cluster.png'); plt.close()
    