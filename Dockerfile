FROM python:3.7
ENV PYTHONPATH "${PYTHONPATH}:/src"

WORKDIR /src

COPY app /src/app
COPY core /src/core

RUN pip install -r requirements.txt

WORKDIR /src/app

CMD ["uswgi", "uwsgi.ini"]