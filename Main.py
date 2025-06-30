import base64
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tempfile

# Загрузка переменных окружения из файла .env
load_dotenv()

# Установка API ключа OpenAI в переменные окружения
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Инициализация клиента OpenAI для взаимодействия с API
client = OpenAI()

# Многострочный промт для GPT-4 с инструкциями по анализу медицинских изображений
sample_prompt = """Answer in Russian. You are a medical practictioner and an expert in analzying medical related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".
Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'
Now analyze the image and answer the above questions in the same structured manner defined above."""

# Инициализация состояния для загруженного файла в Streamlit session state
if 'uploadedFile' not in st.session_state:
    st.session_state.uploadedFile = None

# Инициализация состояния для результата анализа
if 'result' not in st.session_state:
    st.session_state.result = None

def encode_img(image_path):
    """Функция для кодирования изображения в base64"""
    # Открытие файла изображения в бинарном режиме
    with open(image_path, 'rb') as file:
        # Кодирование содержимого файла в base64 и декодирование в utf-8 строку
        return base64.b64encode(file.read()).decode('utf-8')

def call_gpt4(filename: str, sample_prompt = sample_prompt):
    """Функция для взаимодействия с GPT-4 Vision API"""
    # Кодирование изображения в base64
    base64_image = encode_img(filename)

    # Отправка запроса к API GPT-4 Vision
    response = (client.chat.completions.create(
        model = "gpt-4o-mini",  # Указание модели
        messages = [
            {
                "role": "user",  # Роль пользователя в диалоге
                "content":[
                    {
                        "type": "text", "text": sample_prompt  # Текстовый промт
                    },
                    {
                        "type": "image_url",  # Тип контента - изображение
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",  # base64 изображения
                            "detail": "high"  # Высокий уровень детализации анализа
                        }
                    }
                ]
            }
        ],
        max_tokens = 250  # Максимальное количество токенов в ответе
    )).choices[0].message.content  # Извлечение текста ответа

    return response

def answer_five_years_old(query):
    """Функция для упрощения текста до уровня 5-летнего ребенка"""
    # Формирование промта для упрощения
    content = "Answer in Russian. You have to explain the below piece of information to a five years old. \n" + query

    # Отправка запроса к GPT-3.5 Turbo
    responses = (client.chat.completions.create(
        model="gpt-3.5-turbo",  # Использование более дешевой модели
        messages=[
            {"role": "user", "content": content}  # Простой запрос
        ],
        max_tokens=250  # Ограничение длины ответа
    )).choices[0].message.content  # Извлечение текста ответа

    return responses

# Создание заголовка приложения в Streamlit
st.title("Медицинская помощь с использованием мультимодального LLM")

# Создание раскрывающейся секции "О приложении"
with st.expander("Об этом приложении"):
    st.write("Загрузите изображение, чтобы получить анализ от GPT-4.")

# Создание загрузчика файлов с указанием поддерживаемых форматов
uploadedFile = st.file_uploader("Загрузить изображение", type=["jpg", "jpeg", "png"])

# Обработка загруженного файла
if uploadedFile is not None:
    # Создание временного файла с тем же расширением, что и оригинал
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploadedFile.name)[1]) as tmp_file:
        # Запись содержимого загруженного файла во временный файл
        tmp_file.write(uploadedFile.getvalue())
        # Сохранение пути к временному файлу в session state
        st.session_state['filename'] = tmp_file.name

    # Отображение загруженного изображения в интерфейсе
    st.image(uploadedFile, caption='Uploaded Image')

# Обработка нажатия кнопки "Анализ изображения"
if st.button('Анализ изображения'):
    # Проверка наличия файла и его существования на диске
    if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
        # Вызов функции анализа и сохранение результата в session state
        st.session_state['result'] = call_gpt4(st.session_state['filename'])
        # Отображение результата с поддержкой HTML-разметки
        st.markdown(st.session_state['result'], unsafe_allow_html=True)
        # Удаление временного файла после анализа
        os.unlink(st.session_state['filename'])

# Проверка наличия результата анализа
if 'result' in st.session_state and st.session_state['result']:
    # Информационное сообщение о доступности упрощенного объяснения
    st.info("Ниже приведен вариант, который можно понять более простыми словами")

    # Создание переключателя для отображения упрощенного объяснения
    if st.radio("Объясни, как будто мне 5 лет", ('No', 'Yes')) == 'Yes':
        # Генерация и отображение упрощенного объяснения
        simplified_explanation = answer_five_years_old(st.session_state['result'])
        st.markdown(simplified_explanation, unsafe_allow_html=True)