import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


model = joblib.load("lr_model.pkl")
vectorizer = TfidfVectorizer()
X_train_preprocessed = joblib.load("X_train_preprocessed.pkl")
vectors = vectorizer.fit_transform(X_train_preprocessed)

def app():
    st.title("Sentiment Analysis on Customer Reviews on Trip Advisor")
    review = st.text_input("Enter your review")

    if st.button("Detect"):
        vect_review = vectorizer.transform([review])
        pred = model.predict(vect_review)[0]
        if pred == "Positive":
            result = "This is a POSITIVE review"
        elif pred == "Negative":
            result = "This is a NEGATIVE review"
        else:
            result = "This is a NEUTRAL review"
        
        st.write(result)
    


if __name__ == "__main__":
    app()