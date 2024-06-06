# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM python:3.10


RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gunicorn

ENV AWS_ACCESS_KEY_ID=your_access_key_id_here
ENV AWS_SECRET_ACCESS_KEY=your_secret_access_key_here

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY CNN /opt/program
WORKDIR /opt/program