import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st

# Load the text file and preprocess the data
with open('file/clean_output.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')
    
# Tokenize the text into sentences
sentences = sent_tokenize(data)

nltk.download('wordnet')
stemmer = SnowballStemmer('english')

# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    tokenizer = RegexpTokenizer(r'\w+')
    
    words = tokenizer.tokenize(sentence)

    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word not in stopwords.words('english') and word not in string.punctuation]
    
    #stemmed_words = [stemmer.stem(word) for word in words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence

def chatbot(question):
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    # Return the answer
    return most_relevant_sentence

st.title("Chatbot")
st.write("Hello! I'm a chatbot. Ask me anything about Sherlock Holmes' Adventures.")
# Get the user's question
question = st.text_input("Enter a question:")
# Create a button to submit the question
if st.button("Submit"):
    # Call the chatbot function with the question and display the response
    response = chatbot(question)
    st.success(response)
