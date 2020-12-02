FROM python:3.8


ENV WORK_DIR=/home
ENV USER=app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN useradd -m $USER
USER $USER


RUN pip install --upgrade pip

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY ./topic_thunder ./topic_thunder
COPY ./README.md ./README.md


RUN chown -R $USER:$USER $APP_HOME

CMD ["uvicorn", "topic_thunder.api:app", "--host", "0.0.0.0", "--port", "8080"]

#COPY model.zip ./model.zip
#RUN unzip model.zip
#RUN mkdir model
#RUN mv content/model/* model/
#RUN rm -r content
#RUN rm model.zip