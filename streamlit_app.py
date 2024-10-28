# prompt: tambhakan code untuk tampilan interface  inputan dan hasilnya AI untuk di running pada streamlit 

import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import pickle
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
from spacy import displacy
import warnings
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from transformers import pipeline
import pandas as pd

nlp = spacy.load('en_core_web_sm')
warnings.filterwarnings("ignore", category=DeprecationWarning)

news = pd.read_csv('../train_data_cleaning.csv', encoding='latin1') 
print(news)

text = news
documents = news['text'].tolist()
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(documents, show_progress_bar=False)


def tokenize(news):
  '''
  Performs tokenization of input documents.
  Args:
    - docs: list of strings with input documents
  Output:
    - list of strings with tokenized input
  '''
  tokenized_docs = []
  for doc in news:
    # Convert doc to string if it's not already
    if not isinstance(doc, str):
      doc = ' '.join(map(str, doc))  # Join list elements into a string
    
    tokens = gensim.utils.simple_preprocess(doc, deacc=True)
    tokenized_docs.append(tokens)
  return (tokenized_docs)
print('ok')

docs = news.values.tolist()
tokenized_docs = tokenize(docs)
print("ok")
id2word = corpora.Dictionary(tokenized_docs)
print('ok')
len(id2word)


corpus = []
for doc in tokenized_docs:
    corpus.append(id2word.doc2bow(doc))
print('ok')

umap_model = UMAP(n_neighbors=20,
                  n_components=3,
                  min_dist=0.5,
                  metric="euclidean",
                  random_state=120)
hdbscan_model = HDBSCAN(min_cluster_size=15,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        prediction_data=True)
vectorizer_model = CountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer()


model = BERTopic(language='indonesian',
                nr_topics=10,
                embedding_model=sentence_model,           # Step 1 - Extract embeddings
                umap_model=umap_model,                    # Step 2 - Reduce dimensionality
                hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
                vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
                ctfidf_model=ctfidf_model)                # Step 5 - Extract topic words

topics, probs = model.fit_transform(documents, embeddings=embeddings)

# prompt: tampilkan top topic
freq = model.get_topic_info();
print(freq.head(10))

model.get_topic_info()

# prompt: ambil setiap topik dan tampilkan, tampilkan 5 dokumen teratas, serta buatkan summarizenya

# Assuming 'model' and 'documents' are defined from the previous code
for topic_id in range(model.get_topic_info().shape[0] - 1):  # Exclude the "-1" topic (Outliers)
  print(f"\nTopic {topic_id}:")
  topic_words = model.get_topic(topic_id)
  if topic_words:
    print("Top words:", ", ".join([word[0] for word in topic_words]))
  # Get the top 5 documents for the topic
  topic_documents = [documents[i] for i, t in enumerate(topics) if t == topic_id]
  if topic_documents:
    print("\nTop 5 Documents:")
    for i, doc in enumerate(topic_documents[:5]):
      print(f"{i+1}. {doc}")
    # Summarize the top 5 documents
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary_input = "\n".join(topic_documents[:5])  # Combine top 5 documents
    try:
      summary = summarizer(summary_input, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
      print("\nSummary:")
      print(summary)
    except:
      print("\nSummarization failed for this topic.")

new_text = "that"
num_of_topics = 10
similiarity = model.find_topics(new_text, top_n=num_of_topics);
pprint(f'the top {num_of_topics} similiar topics are {similiarity} and similiarities are {np.round(similiarity,2)}')

for i in range(len(similiarity[0])):  # Iterate over the length of the topic indices list
    pprint(f'the top keyword for topic {similiarity} are : ')
    pprint(model.get_topic(similiarity[0][i]))

new_text = "that"
num_of_topics = 10
similarity = model.find_topics(new_text, top_n=num_of_topics)
pprint(f'Top {len(similarity[0])} topik yang mirip adalah {similarity[0]} dengan skor kesamaan {np.round(similarity[1], 2)}')
#Change the range of the loop to match the number of found topics:
for i in range(len(similarity[0])):  
    topic_id = similarity[0][i]  # ID topik
    print("\n")
    pprint(f'Topik {topic_id} memiliki kata kunci: ')
    pprint(model.get_topic(topic_id))


# Streamlit app
st.title("Topic Modeling Interface")

# Input text area
input_text = st.text_area("Enter text here...", "")

# Number of topics slider
num_topics = st.slider("Number of Topics:", 1, 20, 10)

if st.button("Analyze"):
    if input_text:
        similarity = model.find_topics(input_text, top_n=num_topics)
        st.write(f"Top {len(similarity[0])} similar topics are {similarity[0]} with similarity scores {np.round(similarity[1], 2)}")
        for i in range(len(similarity[0])):
            topic_id = similarity[0][i]
            st.write(f"\nTopic {topic_id} has keywords: ")
            st.write(model.get_topic(topic_id))
    else:
        st.warning("Please enter some text.")
