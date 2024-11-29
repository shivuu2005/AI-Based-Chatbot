import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from googletrans import Translator

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and translator
lemmatizer = WordNetLemmatizer()
translator = Translator()

# Load necessary files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# User preferences
user_preferences = {"preferred_language": "Hinglish"}  # Default language preference

def clean_up_sentence(sentence):
    """  
    Tokenize and lemmatize a sentence.
    """
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    """
    Create a bag-of-words representation for a sentence.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """
    Predict the class (intent) of a user's sentence.
    """
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    """
    Get a random response based on the predicted intent.
    """
    if not intents_list:
        return "I'm sorry, I didn't understand that."
    
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    
    return "I'm sorry, I didn't understand that."

def translate_to_english(hinglish_response):
    """
    Translate Hinglish response to English.
    """
    return translator.translate(hinglish_response, src='hi', dest='en').text

def chatbot_response(message):
    """
    Generate a chatbot response for a given message.
    """
    global user_preferences

    # Check if user wants to switch language preference
    if "i prefer english" in message.lower() or "english please" in message.lower():
        user_preferences["preferred_language"] = "English"
        return "Alright, I will respond in English from now on."
    elif "mujhe hindi me baat karni hai" in message.lower() or "hindi please" in message.lower() or "hindi" in message.lower():
        user_preferences["preferred_language"] = "Hinglish"
        return "Thik hai mai hindi me bat krta hu"

    # Predict intent and get response
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)

    # Translate response to English if preferred
    if user_preferences["preferred_language"] == "English":
        response = translate_to_english(response)

    return response

if __name__ == "__main__":
    print("|============= Welcome to DSS Chatbot =============|")
    print("|======================= Feel Free ========================|")
    print("|========================== To ============================|")
    print("|====== Ask any query about our Website ======|")
    print("| Type 'bye' or 'goodbye' to exit the chat. |")

    while True:
        # Input from the user
        message = input("| You: ").strip()

        # Exit condition
        if message.lower() in ["bye", "goodbye", "exit"]:
            print("| Bot: Goodbye! Have a great day!")
            print("|===================== Program Ended =====================|")
            break

        # Generate and print chatbot response
        response = chatbot_response(message)
        print(f"| Bot: {response}")
