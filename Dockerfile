FROM python:3.7
ENV PYTHONPATH "${PYTHONPATH}:/src"

WORKDIR /src

COPY app /src/app
COPY core /src/core

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /src/app

CMD ["uswgi", "uwsgi.ini"]