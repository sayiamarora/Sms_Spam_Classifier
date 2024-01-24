import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title and description with color
st.title("📧 Email/SMS Spam Classifier")
st.write(
    "This app uses a machine learning model to classify whether an input message is spam or not."
)

# Input text area with color
input_sms = st.text_area("Enter the message", key="message_input", height=150, max_chars=500)
input_sms_placeholder = st.empty()

# Prediction button with a colorful background
if st.button('Predict', key="predict_button", help="Click to make a prediction"):
    input_sms_placeholder.empty()  # Clear the input placeholder
    st.info("🔄 Predicting...")  # Display a loading message

    # 1. Preprocess the input message
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize the transformed message
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict using the loaded model
    result = model.predict(vector_input)[0]

    # Display the prediction result with colored feedback
    if result == 1:
        st.error("🚨 Spam Detected! You might want to be cautious.")
    else:
        st.success("✔️ Not Spam! Your message seems safe.")
