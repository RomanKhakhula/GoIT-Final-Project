# Docker-команда FROM вказує базовий образ контейнера
FROM python:3.12

# Встановимо змінну середовища
ENV APP_HOME /app

# Встановимо робочу директорію всередині контейнера
WORKDIR $APP_HOME

# Встановимо залежності всередині контейнера
COPY requirements.txt $APP_HOME/requirements.txt

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Скопіюємо інші файли в робочу директорію контейнера
COPY . .

# Позначимо порт, де працює застосунок всередині контейнера
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Запустимо наш застосунок всередині контейнера
ENTRYPOINT ["streamlit", "run", "Final_GoIT_Project.py", "--server.port=8501", "--server.address=0.0.0.0"]
