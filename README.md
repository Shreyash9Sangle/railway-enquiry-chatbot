# Railway Chatbot

## Overview
The **Railway Ticketing Enquiry Chatbot** is an AI-powered system that provides users with railway-related information, such as train availability, timings, and other ticketing queries.  
It integrates **Natural Language Processing (NLP), Deep Learning, and Speech Recognition** to offer an interactive user experience. The project also features a simple **Gradio-based UI** for easy interaction.

## Features
- Understands and processes **natural language queries** about trains.
- Provides **train availability and schedule information** using a structured dataset.
- **Speech-to-Text Input**: Users can speak queries to the chatbot.
- **Text-to-Speech Output**: The bot can speak responses aloud.
- Built with **Deep Learning-based intent classification** for accurate query handling.
- Interactive **Gradio UI** for easy access and usage.

## Project Structure
├── Train_data_log_0.csv/ # Dataset containing train information

├── speech_chatbot_data.pkl/ # Trained deep learning model

├── Final_UI_version.py # Main chatbot script

├── requirements.txt # Python dependencies

└── README.md # Project documentation


## Tech Stack
- **Python**  
- **TensorFlow** (for deep learning model)  
- **NLTK** (for NLP preprocessing)  
- **SpeechRecognition** (for voice input)  
- **pyttsx3** (for text-to-speech output)  
- **Gradio** (for UI interface)  

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/railway-chatbot.git
cd railway-chatbot
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Usage:
 Run the chatbot in the console- 
"python speech_recognition_chatbot.py"

## Results

Successfully handles most train-routes related queries with accurate responses.

Integrates speech recognition + NLP + DL + UI into a single functional system.

## Future Improvements

Integrate real-time IRCTC API for live train availability and booking.

Expand chatbot to handle PNR status and ticket booking & cancellations.

Add a web deployment option for broader accessibility.

## Author
Shreyash Sangle
