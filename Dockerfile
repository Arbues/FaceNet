FROM tensorflow/tensorflow:latest-gpu

ENV TF_FORCE_GPU_ALLOW_GROWTH=true


COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY modelo .

CMD ["python", "main.py"]
