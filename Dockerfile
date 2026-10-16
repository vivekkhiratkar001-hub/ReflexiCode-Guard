# Reproducible runner (26 Oct demo backup). NOT built/tested in the authoring sandbox - build once on your machine:
#   docker build -t reflexicode-guard . && docker run --rm reflexicode-guard
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
ENV PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
CMD ["./scripts/regression.sh"]
