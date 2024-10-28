import streamlit as st
import pandas as pd
from bertopic import BERTopic
import nltk

# Pastikan untuk mengunduh stopwords jika belum
nltk.download('stopwords') 

# Fungsi untuk memproses teks dan menghasilkan topik
def generate_topics(text, n_topics=5):
    # Membagi teks menjadi daftar kalimat
    documents = text.split('\n')
    
    # Menggunakan BERTopic untuk menemukan topik
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(documents)
    
    # Mengambil topik dan artikel teratas
    topic_info = topic_model.get_topic_info()
    top_topics = topic_info.head(n_topics)
    
    return top_topics, topics

# Antarmuka Streamlit
st.title("Model BERTopic dengan Streamlit")

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks (pisahkan dengan baris baru):", height=300)

# Tombol untuk memproses input
if st.button("Proses Teks"):
    if user_input:
        # Menghasilkan topik
        top_topics, topics = generate_topics(user_input)
        
        # Menampilkan daftar topik
        st.subheader("Daftar Topik")
        st.write(top_topics)
        
        # Menampilkan artikel per topik
        for index, row in top_topics.iterrows():
            st.subheader(f"Topik {row['Topic']}: {row['Name']}")
            articles = [user_input.split('\n')[i] for i in range(len(topics)) if topics[i] == row['Topic']]
            st.write(articles)
    else:
        st.warning("Silakan masukkan teks untuk diproses.")
