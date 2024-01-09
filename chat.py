import pyttsx3
import speech_recognition as sr 

import json
import random

import torch
from model1 import NeuralNet
from nltk_utils import bag_of_words, tokenize


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  
engine.setProperty('rate', 170)
engine.setProperty('volume',1.0) 

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def recognizer():  
    try: 
        r = sr.Recognizer()
        with sr.Microphone() as voice:
            r.adjust_for_ambient_noise(voice, duration=.2) 
            speak("listening")
            audio = r.listen(voice,phrase_time_limit=15)
            MyText = r.recognize_google(audio,language="en-in")
            MyText = MyText.lower()
        return MyText
    except Exception as e:
        # speak("voice not found...")
        return " "
    except KeyboardInterrupt:
        pass
    
    
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# bot_name = "____"
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

while True:
    sentence = recognizer()
    if "stop" in sentence :
        break
    resp = get_response(sentence)
    speak(resp)
        