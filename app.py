import streamlit as st
import joblib

# Load saved files
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📩 Spam Detection App")

message = st.text_area("Enter a message")

if st.button("Predict"):

    msg_vector = vectorizer.transform([message])

    prediction = model.predict(msg_vector)

    probability = model.predict_proba(msg_vector)

    if prediction[0] == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Ham Message")

    st.write(
        f"Spam Probability: {probability[0][1]:.2%}"
    )