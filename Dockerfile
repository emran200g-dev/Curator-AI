FROM python:3.11-slim

WORKDIR /app

# Copy requirements from your specific subfolder
COPY ./NEW_UI/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend files from the subfolder
COPY ./NEW_UI/backend/ .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--log-level", "info"]
