FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/api/ ./dashboard/api/
COPY data/raw/models/ ./data/raw/models/

EXPOSE 8001

CMD ["uvicorn", "dashboard.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
