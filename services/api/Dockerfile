FROM python:3.11-slim

WORKDIR /app

# context is project root (.)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] pretty_midi mido

# copy app and src from project root
COPY services/api/app.py /app/app.py
COPY src /app/src

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
