FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /

COPY requirements.txt /
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "./train.py"]