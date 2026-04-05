FROM python:3.11-slim

EXPOSE 8000

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD [ "gunicorn", "-b", "0.0.0.0:8000", "wsgi:app" ]