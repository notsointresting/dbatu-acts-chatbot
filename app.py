from flask import Flask
from flask import request
from flask import Response
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
from main import user_input

load_dotenv()

# Give OpenAI Key
genai.configure(api_key=os.getenv("API_KEY"))

app = Flask(__name__)
# Get BOT Token from telegram
token = os.getenv("TELEGRAM_BOT_TOKEN")


def generate_answer(question):
    answer = user_input(question)['output_text']
    return answer



# To Get Chat ID and message which is sent by client
def message_parser(message):
    chat_id = message['message']['chat']['id']
    text = message['message']['text']
    print("Chat ID: ", chat_id)
    print("Message: ", text)
    return chat_id, text


# To send message using "SendMessage" API
def send_message_telegram(chat_id, text):
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': text
    }
    response = requests.post(url, json=payload)
    return response


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        msg = request.get_json()
        chat_id, incoming_que = message_parser(msg)
        answer = generate_answer(incoming_que)
        send_message_telegram(chat_id, answer)
        return Response('ok', status=200)
    else:
        return "<h1>Something went wrong</h1>"



