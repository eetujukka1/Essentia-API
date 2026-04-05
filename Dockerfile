FROM python:3.11-slim

EXPOSE 3000

WORKDIR /usr/src/app

COPY . .

RUN pip install -r requirements.txt

CMD [ "gunicorn", "wsgi:app" ]