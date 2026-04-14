FROM python:3.10-slim

WORKDIR /life_expectancy

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY src ./src
