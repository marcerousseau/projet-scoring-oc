# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM --platform=linux/amd64 python:3.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

#Set the configuration to use when deployed
ENV CONFIGMODE=production

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR $APP_HOME/app

ENV PORT 8080

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 wsgi:app
