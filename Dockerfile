FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --upgrade pip
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables and expose port
ENV PORT=8080
ENV WORKERS=1
EXPOSE $PORT

CMD gunicorn --bind 0.0.0.0:$PORT --workers $WORKERS --timeout 0 main:app -k uvicorn.workers.UvicornWorker