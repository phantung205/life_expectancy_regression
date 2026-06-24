FROM python:3.10-slim

WORKDIR /life_expectancy

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY src ./src
COPY app.py ./app.py
COPY validation ./validation
COPY templates ./templates

EXPOSE 5000

CMD ["python", "app.py"]
