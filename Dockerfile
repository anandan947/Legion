FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PATH="/usr/local/bin:${PATH}"

EXPOSE 8000

CMD ["pytest", "-v"]
