import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize,stem

import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)
def talk(text):
    engine.say(text)
    engine.runAndWait()

with open('intents.json',encoding='utf8') as f:
    intents = json.load(f)
engine = pyttsx3.init()

FILE = 'data.pth'
data = torch.load(FILE)

model_state=data["model_state"]
output_size=data["output_size"]
input_size=data["input_size"]
hidden_size=data["hidden_size"]
tags=data["tags"]
all_words=data["all_words"]
     
model = NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()


#final implementation
with open('intents.json', encoding='utf8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def chat(msg):
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
    
    return "whippeeeee"


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = chat(sentence)
        print(resp)

                
                    


