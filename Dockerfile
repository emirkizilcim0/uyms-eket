FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y curl


RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pillow google-generativeai playwright
RUN playwright install
RUN pip install -e .


CMD ["tail", "-f", "/dev/null"]