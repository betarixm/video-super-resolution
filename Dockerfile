FROM python:3.7
ENV PYTHONPATH "${PYTHONPATH}:/src"

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

WORKDIR /src

COPY app /src/app
COPY core /src/core

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /src/app

CMD ["uwsgi", "uwsgi.ini"]