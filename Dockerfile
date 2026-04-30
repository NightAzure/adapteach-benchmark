FROM python:3.12-slim

WORKDIR /app

COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

COPY . /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "bench/run_obj1.py", "--benchmark", "bench/benchmarks/obj1_primary.yaml", "--sample-per-dataset", "5"]
