FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update && apt-get -y upgrade
RUN apt -y install build-essential
ADD Docker/requirements.txt ./
RUN pip install -r requirements.txt
