FROM python:3.11-slim

WORKDIR /app
# add python-multipart here
RUN pip install --no-cache-dir fastapi uvicorn jinja2 requests python-multipart

COPY main.py /app/main.py
COPY templates /app/templates
ENV API_URL=http://localhost:8000
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
