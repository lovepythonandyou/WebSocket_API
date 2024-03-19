import csv
import json
import pandas as pd


from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from datetime import datetime
from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from starlette.templating import Jinja2Templates

load_dotenv()

app = FastAPI()


class CSVFileHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None

    def __enter__(self):
        self.file = open(self.file_path, 'a', newline='')
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()


@app.on_event("startup")
async def startup_event():
    global csv_file
    with CSVFileHandler('chat_log.csv') as file:
        csv_file = file


@app.on_event("shutdown")
async def shutdown_event():
    if csv_file:
        csv_file.close()


templates = Jinja2Templates(directory="templates")
llm = ChatOpenAI()
output_parser = StrOutputParser()


# Функция для записи в буффер ключ-значений
def update_memory_from_csv(memory, file_path):
    # Чтение данных из файла CSV
    df = pd.read_csv(file_path, encoding='cp1251')

    # Выбор последних трех строк
    last_three_rows = df.tail(3)

    # Проход по каждой строке и обновление памяти
    for index, row in last_three_rows.iterrows():
        input_message = row['сообщение']
        output_message = row['ответ']
        memory.save_context({"input": input_message}, {"output": output_message})


def get_last_message_history():
    responses = []

    # Открываем файл и считываем строки
    with open('chat_log.csv', 'r', encoding='cp1251') as file:
        reader = csv.DictReader(file)

        # Проходимся по строкам в обратном порядке и добавляем значения столбца "ответ"
        for row in reversed(list(reader)):
            responses.append(row['ответ'])
            if len(responses) == 3:
                break

    # Соединяем последние ответы в один текст
    last_message = '\n'.join(responses[::-1])
    return last_message


res_mes = get_last_message_history()
# PROMPT = ChatPromptTemplate.from_messages([
#     ("system", res_mes),
# ])


# daily_context = res_mes
#
# template = """
# Here is some context about today: {daily_context}
# Current conversation:
# {history}
# user: {input}
# AI:"""
# PROMPT = PromptTemplate.from_template(template).partial(daily_context=daily_context)
#
# conversation = ConversationChain(
#     prompt=PROMPT,
#     llm=llm,
#     verbose=True,
#     memory=ConversationBufferWindowMemory(k=3)
# )


cbwm = ConversationBufferWindowMemory(k=3, return_messages=True)
update_memory_from_csv(cbwm, 'chat_log.csv')
template = """
Current conversation:
{history}
user: {input}
AI:
"""
PROMPT = PromptTemplate.from_template(template).partial()

conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=cbwm
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
    writer = csv.writer(csv_file)
    writer.writerow([timestamp, message, result])


# Функция для генерации ответа
def get_answer(message):
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

            res_mes = get_last_message_history()
            await manager.broadcast(f"INFO: {res_mes}")

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
