FROM bitnami/pytorch:latest
LABEL maintainer "yunsunset <uoon97@gmail.com>"
# RUN apt-get update
# RUN apt-get -yq install python3 python3-pip
COPY app /app
COPY app/requirements.txt /app
WORKDIR /app
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3" ]
CMD ["model_2.py"]

