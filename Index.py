import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load TF-IDF Vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load tf idfvectorizer

# Load LDA Model
lda_mod = joblib.load('svm_lda.pkl')  # Load model LDA

# Load Best Classifier Model
best_classifier_model = joblib.load('svm_model.pkl')  # Model terbaik

# Main Streamlit app
def main():
    st.title('Klasifikasi Berita online')

    # Input Text Area
    input_text = st.text_area('Masukkan berita yang ingin klasifikasi:','')

    if st.button('Klasifikasi'):
        # Transformasi TF-IDF
        vectorized_input = tfidf_vectorizer.transform([input_text])

        # Transformasi LDA
        input_lda = lda_mod.transform(vectorized_input)

        # Prediksi menggunakan model terbaik
        prediction = best_classifier_model.predict(input_lda)

        # Menampilkan hasil prediksi
        st.write(f'Hasil Klasifikasi: {prediction[0]}')

if __name__ == '__main__':
    main()
