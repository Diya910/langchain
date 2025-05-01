from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

while True:
    user_input = input('You:')
    if user_input.lower() == 'exit':
        break
    response = model.invoke(user_input)
    print(f"Chatbot: {response.content}")