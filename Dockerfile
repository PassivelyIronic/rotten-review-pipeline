# Gradio demo image. Assumes trained artifacts/ are mounted or baked in:
#   docker build -t rotten-review .
#   docker run -p 7860:7860 -v $(pwd)/artifacts:/app/artifacts rotten-review
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[app]"

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
CMD ["python", "-m", "rotten_review.app"]
