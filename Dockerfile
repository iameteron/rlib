FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app

COPY . /app

RUN rm -rf /etc/apt/sources.list.d/* && \
    sed -i '/developer.download.nvidia.com/d' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    ssh \
    && apt-get clean

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["service", "ssh", "start"]