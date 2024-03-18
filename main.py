import csv
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from datetime import datetime
from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from starlette.templating import Jinja2Templates

load_dotenv()
app = FastAPI()
templates = Jinja2Templates(directory="templates")
llm = ChatOpenAI()
output_parser = StrOutputParser()

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=10)
)


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


# Функция для записи данных в CSV файл
def write_to_csv(timestamp, message, result):
    with open('chat_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, message, result])


# Функция для генерации ответа
def get_answer(message):
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system",
    #      "Вы самый умный человек"),
    #     ("user", "{ввод}")
    # ])
    #
    # chain = prompt | llm | output_parser

    # result = chain.invoke({"ввод": message})
    return conversation.predict(input=message)


# Возвращаем страницу html
@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            # Запоминаем текст и время отправки сообщений
            data = await websocket.receive_text()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Выводим в чат сообщение
            await manager.broadcast(f"Client #{client_id} says: {data}")

            # Запоминаем ответ, время и переводим их в формат json
            result = get_answer(data)
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = {"time": time, "result": result}
            json_message = json.dumps(message, ensure_ascii=False)
            # Выводим в чат сообщение с ответом
            await manager.broadcast(f"OpenAI says: {json_message}")

            # Записываем время отправки, вопрос, ответ в csv файл
            write_to_csv(timestamp, data, result)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")
