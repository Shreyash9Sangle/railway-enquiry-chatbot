import gradio as gr
import speech_recognition as sr
import pyttsx3
import pandas as pd
import json
import random
import pickle
import numpy as np
import threading
import time
import os
import signal
from tensorflow.keras.models import load_model
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Initialize Stemmer
stemmer = LancasterStemmer()
r = sr.Recognizer()

# Load intents file
with open('intents.json') as jd:
    intents = json.load(jd)

# Load train data CSV
train_df = pd.read_csv("Train_data_log_0.csv")

# Load trained chatbot model
model = load_model('speech_chatbot_model.h5')
with open('speech_chatbot_data.pkl', 'rb') as f:
    words, classes, training, output = pickle.load(f)

def bot_speaking(message):
    def speak():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(message)
        engine.runAndWait()
    threading.Thread(target=speak, daemon=True).start()

def exit_chatbot():
    bot_speaking("Goodbye! Chatbot is shutting down.")
    threading.Thread(target=delayed_exit, daemon=True).start()
    return "Goodbye! Chatbot is shutting down."

def delayed_exit():
    time.sleep(3)  # Give time for speaking
    os.kill(os.getpid(), signal.SIGTERM)

def find_trains(source, destination):
    source, destination = source.lower().strip(), destination.lower().strip()
    results = train_df[(train_df['Source'].str.lower().str.strip() == source) &
                       (train_df['Destination'].str.lower().str.strip() == destination)]

    if results.empty:
        response = "No trains available for this route."
        bot_speaking(response)
        return response

    response = "Here are the available trains:\n"
    full_speech = "Here are the available trains."

    for _, row in results.iterrows():
        train_info = (
            f"Train Name: {row['Train Name']}, Departure Station: {row['Departure Station']}, "
            f"Destination Station: {row['Destination Station']}, Departure Time: {row['Departure Time']}"
        )
        response += train_info + "\n"
        full_speech += " " + train_info + "."

    bot_speaking(full_speech)
    return response

def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def classify(sentence):
    results = model.predict(np.array([bag_of_words(sentence, words)]))[0]
    results = [[i, r] for i, r in enumerate(results) if r > 0.15]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results] if results else [('unknown', 0)]

def chatbot_response(sentence):
    if sentence.lower() in ["exit", "quit", "goodbye", "bye"]:
        return exit_chatbot()

    results = classify(sentence)
    if results and results[0][0] != 'unknown':
        if results[0][0] == 'check_train_schedule':
            words = sentence.split()
            try:
                source, destination = words[words.index('from')+1], words[words.index('to')+1]
                return find_trains(source, destination)
            except (ValueError, IndexError):
                response = "Please specify source and destination correctly."
                bot_speaking(response)
                return response
        for i in intents['intents']:
            if i['tag'] == results[0][0]:
                reply = random.choice(i['responses'])
                bot_speaking(reply)
                return reply
    response = "I'm sorry, I didn't understand that."
    bot_speaking(response)
    return response

def voice_input():
    with sr.Microphone() as source:
        try:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio)
            return chatbot_response(text)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Speech recognition service is unavailable."

def chatbot_interface(user_input):
    return chatbot_response(user_input)

def main():
    with gr.Blocks(theme="dark") as demo:
        gr.Markdown("## 🚆 Railway Chatbot with Voice Support")
        gr.Markdown("Ask me about train schedules or general queries!")

        with gr.Row():
            with gr.Column(scale=4):
                text_input = gr.Textbox(label="You:", placeholder="Type your message here...")
                send_button = gr.Button("📧 Send", size="sm")
                voice_button = gr.Button("🎤 Speak", size="sm")
                exit_button = gr.Button("🚪 Exit", size="sm")
            with gr.Column(scale=6):
                output_text = gr.Textbox(label="Chatbot:", interactive=False, lines=8, placeholder="Chatbot responses will appear here...")

        send_button.click(fn=chatbot_interface, inputs=text_input, outputs=output_text)
        voice_button.click(fn=voice_input, inputs=[], outputs=output_text)
        exit_button.click(fn=exit_chatbot, inputs=[], outputs=output_text)

    demo.launch()

if __name__ == "__main__":
    main()
