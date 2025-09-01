import streamlit as st
import pandas as pd
import re
import string
import nltk

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords & lemmatizer resources
# This is necessary for the text cleaning function to work
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# new load_data function
# Load and prepare the data
# Using st.cache_data to cache the data and avoid reloading on every interaction
@st.cache_data
def load_data():
    # Reading the datasets from local files in your repository
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')
    
    # Assigning labels: 1 for fake, 0 for true
    fake['labels'] = 1
    true['labels'] = 0
    
    # Combining and shuffling the data for randomness
    df = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# Clean the text
def clean_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'\d+', '', str(text).lower()) # Remove numbers and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    
    words = text.split()
    # Lemmatize and remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Train the model
# Using st.cache_resource to cache the trained model and vectorizer
@st.cache_resource
def train_model(df):
    df["clean_text"] = df["text"].apply(clean_text)
    x = df["clean_text"]
    y = df["labels"]

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Vectorizing the text data using TF-IDF
    vectorizer = TfidfVectorizer(max_df=0.7)
    x_train_idf = vectorizer.fit_transform(x_train)
    x_test_idf = vectorizer.transform(x_test)

    # Training the Logistic Regression model
    model = LogisticRegression()
    model.fit(x_train_idf, y_train)
    
    # Evaluating the model
    acc = accuracy_score(y_test, model.predict(x_test_idf))
    report = classification_report(y_test, model.predict(x_test_idf), target_names=['True', 'Fake'])

    return model, vectorizer, acc, report

# --- Streamlit User Interface ---

st.title("📰 Fake News Detection App")
st.markdown("Enter a news article or statement below, and the model will predict if it's **Real** or **Fake**.")

# Load data and train the model
df = load_data()
model, vectorizer, acc, report = train_model(df)

# Input from user
user_input = st.text_area("✍️ Enter News Content Here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        # Clean and vectorize the user input
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        
        # Make a prediction
        prediction = model.predict(vectorized)[0]
        label = "Fake News 🛑" if prediction == 1 else "Real News ✅"
        
        st.subheader(f"Prediction: {label}")

st.markdown("---")
st.subheader("📊 Model Performance")
st.text(f"Accuracy: {acc * 100:.2f}%")
with st.expander("See Classification Report"):
    st.text(report)
