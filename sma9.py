# ==========================================
# SOCIAL MEDIA COMPETITOR ANALYSIS (COLAB)
# ==========================================

# Install required libraries (Colab usually has most)
!pip install nltk seaborn scikit-learn --quiet

# ------------------------------------------
# 1. Import Libraries
# ------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from google.colab import files

# Download sentiment model
nltk.download('vader_lexicon')

# ------------------------------------------
# 2. Upload Dataset
# ------------------------------------------
print("Upload your CSV file (social_data.csv)")
uploaded = files.upload()

# Get file name
file_name = list(uploaded.keys())[0]

# Load dataset
df = pd.read_csv(file_name)

print("\nDataset Preview:")
print(df.head())

# ------------------------------------------
# 3. Data Preprocessing
# ------------------------------------------
df['engagement'] = df['likes'] + df['comments'] + df['shares']

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['day'] = df['date'].dt.day_name()

# ------------------------------------------
# 4. Competitor Performance Analysis
# ------------------------------------------
performance = df.groupby('competitor')['engagement'].mean().sort_values(ascending=False)

print("\nAverage Engagement per Competitor:\n")
print(performance)

plt.figure()
performance.plot(kind='bar')
plt.title("Competitor Engagement Comparison")
plt.xlabel("Competitor")
plt.ylabel("Average Engagement")
plt.show()

# ------------------------------------------
# 5. Posting Pattern Analysis
# ------------------------------------------
plt.figure()
sns.countplot(data=df, x='day')
plt.title("Posting Frequency by Day")
plt.xticks(rotation=45)
plt.show()

# ------------------------------------------
# 6. Sentiment Analysis
# ------------------------------------------
sia = SentimentIntensityAnalyzer()

df['sentiment'] = df['post'].astype(str).apply(
    lambda x: sia.polarity_scores(x)['compound']
)

print("\nSentiment Analysis Sample:\n")
print(df[['post', 'sentiment']].head())

# ------------------------------------------
# 7. Clustering (Strategy Analysis)
# ------------------------------------------
features = df[['likes', 'comments', 'shares']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_features)

print("\nCluster Distribution:\n")
print(df['cluster'].value_counts())

# ------------------------------------------
# 8. Cluster Visualization
# ------------------------------------------
plt.figure()
sns.scatterplot(
    x=df['likes'],
    y=df['shares'],
    hue=df['cluster'],
    palette='viridis'
)
plt.title("Post Strategy Clusters")
plt.show()

# ------------------------------------------
# 9. Top Performing Posts
# ------------------------------------------
top_posts = df.sort_values(by='engagement', ascending=False).head(5)

print("\nTop Performing Posts:\n")
print(top_posts[['competitor', 'post', 'engagement']])

# ------------------------------------------
# 10. Save Output File
# ------------------------------------------
output_file = "analyzed_social_data.csv"
df.to_csv(output_file, index=False)

print(f"\nAnalysis complete. File saved as {output_file}")

# Download result
files.download(output_file)