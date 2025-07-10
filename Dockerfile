FROM python:3.11-slim

ENV HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="${HOME}/bin:$PATH"
WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main

COPY . .

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
