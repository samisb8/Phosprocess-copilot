FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./

RUN python -c "import pathlib, tomllib; data = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); pathlib.Path('/tmp/requirements.txt').write_text('\n'.join(data['project']['dependencies']) + '\n', encoding='utf-8')" \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY src ./src

ENV PYTHONPATH=/app/src

COPY configs ./configs
COPY data/evaluation/retrieval/v0.1/frozen/dev_best_v3 ./data/evaluation/retrieval/v0.1/frozen/dev_best_v3

COPY alembic.ini ./
COPY migrations ./migrations

EXPOSE 8000

CMD ["uvicorn", "phosprocess.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
