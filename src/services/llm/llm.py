from openai import OpenAI
import requests
from src.services.logging.logger import logger
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


client = OpenAI(
    api_key=OPENAI_API_KEY
)

def ask_openai(prompt: str, model: str = 'gpt-4.1'):

    response = client.chat.completions.create(
        model = model,
        messages=[
        {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def ask_mistral(prompt: str):

    model_name: str = 'mistral:instruct'
    base_url: str = 'http://localhost:11434'

    prompt = prompt

    response = requests.post(
        url=f'{base_url}/api/generate',
        json={
            'model': model_name,
            'prompt': prompt,
            'stream': False
        }
    )
    if response.status_code == 200:
        logger.info(f"[Mistral] Sending prompt:\n{prompt[:300]}...\n")
        return response.json().get('response', '').strip()
    else:
        raise RuntimeError(f'Ollama error {response.status_code}: {response.text}')

prompt = 'How are you?'

print(ask_openai(prompt))

def run(model: str):
    """_summary_

    Args:
        model (str): Choose a model between 'OpenAI' and 'Mistral'
    """
    if model == 'OpenAI':
        ask_openai(prompt)
    elif model == 'Mistral':
        ask_mistral(prompt)
    else:
        logger.info('Choose a model between OpenAI and Mistral')