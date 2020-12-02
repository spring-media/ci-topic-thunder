FROM python:3.8


ENV WORK_DIR=/home
#ENV USER=app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR $WORK_DIR
#RUN useradd -m $USER
#USER $USER


RUN pip install --upgrade pip
RUN pip install --no-cache-dir uvicorn gunicorn

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY ./topic_thunder ./topic_thunder
COPY ./bin ./bin

COPY ./README.md ./README.md

#ENV PYTHONPATH=/app

#RUN chown -R $USER:$USER $APP_HOME

CMD ["python","./topic_thunder/api.py"]
#CMD [ "uvicorn", "topic_thunder.api:app", "--host", "0.0.0.0", "--port", "8080"]
#RUN echo $PATH
#ENTRYPOINT [ "bash", "./.local/bin/start_server" ]

#COPY model.zip ./model.zip
#RUN unzip model.zip
#RUN mkdir model
#RUN mv content/model/* model/
#RUN rm -r content
#RUN rm model.zip