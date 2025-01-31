# Use Ubuntu as the base image
FROM ubuntu:latest

# Set environment variables
ENV WHISPER_HOST=0.0.0.0
ENV WHISPER_PORT=8080

# Update and install necessary dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Whisper package
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE $WHISPER_PORT

# Command to run your application
CMD ["python3", "whisper_server.py"]