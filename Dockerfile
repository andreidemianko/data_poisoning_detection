FROM python:3.13-slim

# Node.js нужен для promptfoo (опциональный детектор prompt-инъекций)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g promptfoo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . .

# Папки для данных и результатов создаём заранее — пользователь монтирует их снаружи
RUN mkdir -p data models reports

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
