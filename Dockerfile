FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
&& pip install --no-cache-dir packaging \
&& pip install --no-cache-dir poetry==1.7.1

RUN poetry config virtualenvs.create false

COPY . .

RUN poetry install --no-interaction --no-ansi

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8000

CMD ["poetry", "run", "pytest", "-v"]
