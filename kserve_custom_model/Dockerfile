# Build an image that can serve mlflow models.
FROM python:3.11.0

RUN apt-get -y update && apt-get install -y --no-install-recommends nginx


WORKDIR /app

# Install Dependencies
COPY ./model_dir/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model to image
COPY ./model_dir /app/model_dir
COPY ./model.py /app/model.py

#RUN chmod o+rwX /opt/mlflow/
#RUN chgrp -R 0 /opt && chmod -R g=u /opt

# clean up apt cache to reduce image size
#RUN rm -rf /var/lib/apt/lists/*

#ENTRYPOINT ["python", "-c", "from mlflow.models import container as C; C._serve('local')"]

ENTRYPOINT [ "python", "-m", "model"]