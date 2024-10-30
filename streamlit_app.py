import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Title and description of the app
st.title("Sentence Search and Clustering with Sentence-BERT")
st.write(""" This app allows you to search for sentences that closely match your input from a
list of predefined sentences using Sentence-BERT. Each result links to further
details about the document.
Additionally, you can cluster sentences into different groups based on their
semantic similarity and visualize the clusters.""")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
  # Load sentences from a CSV file with error handling
  @st.cache_data

  def load_sentences(file):
    try:
    # Attempt to load the CSV, adjusting the delimiter and skipping bad lines
     df = pd.read_csv(file, delimiter=',', on_bad_lines='skip')
     df.columns = df.columns.str.strip() # Strip whitespace from colum names
     return df
    except Exception as e:
       st.error(f"Error reading CSV file: {e}")
       return None

# Load sentences
df = load_sentences(uploaded_file)
if df is not None:
  # Choose the column to use for text
  text_column = st.selectbox("Select the column to use for text", df.columns)
  # Filter rows with non-null values in selected column
  df = df[df[text_column].notna()]
  # Convert to list of dictionaries for compatibility with existing code
  sentences = df.to_dict(orient='records')
  # Load Sentence-BERT model
  
  @st.cache_resource
  def load_model():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model
  
  model = load_model()

  # Generate embeddings and cache the results
  @st.cache_data
  def generate_embeddings(_model, sentences):
    embeddings = _model.encode(sentences)
    return embeddings

# Extract embeddings for each sentence
  embeddings = generate_embeddings(model, [s[text_column] for s in sentences])
# Input query from user
  query = st.text_input("Enter your search query:")
# Add a search button
  if st.button("Search"):
    if query:
# Generate embedding for query
      query_embedding = model.encode([query])
# Compute cosine similarity between query and sentences
      cosine_similarities = cosine_similarity(query_embedding,embeddings)

      similarity_scores = cosine_similarities[0]
# Sort and select top 5 results
      sorted_indices = np.argsort(similarity_scores)[::-1]

      top_5_indices = sorted_indices[:5]
# Prepare and display results
      st.subheader("Search Results:")
      results_found = False
      for index in top_5_indices:
        score = similarity_scores[index]
        if score > 0:
          sentence = sentences[index]
          with st.expander(f"Document {index + 1}: (Similarity Score: {score:.2f})", expanded=False): st.write(f"[{sentence[text_column]}]")
          results_found = True

      if not results_found:
          st.write("No matching results found for your query.")
    else:
      st.warning("Please enter a query before searching.")

# Clustering section
st.subheader("Cluster Sentences with K-Means")
# Slider to select the number of clusters
num_clusters = st.slider("Select number of clusters for K-Means:",min_value=2, max_value=10, value=5)

# Cluster the embeddings using K-Means
def cluster_sentences(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

# Perform clustering and display results
if st.button("Cluster Sentences"): 
    cluster_labels = cluster_sentences(embeddings, num_clusters)
# Add cluster labels to each sentence in the DataFrame
    df['cluster'] = cluster_labels
# Reduce dimensions for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
# Plotting the clusters
    plt.figure(figsize=(10, 6))
    for i in range(num_clusters): 
        plt.scatter(reduced_embeddings[df['cluster'] == i, 0],
                    reduced_embeddings[df['cluster'] == i, 1],
                    label=f'Cluster {i + 1}')
        
plt.title('Sentence Clustering Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Display the plot in Streamlit
st.pyplot(plt)

st.subheader("Clustering Results:")
for cluster in range(num_clusters): 
  st.write(f"\n### Cluster {cluster + 1}")
  cluster_sentences = df[df['cluster'] == cluster][text_column].values

  for sentence in cluster_sentences[:5]: # Display up to 5 sentences per cluster
      st.write(f"- {sentence}")
  else:
    st.warning("Unable to load the CSV file. Please check the file format.")
else:
  st.warning("Please upload a CSV file to proceed.")
