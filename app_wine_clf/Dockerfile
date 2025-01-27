# Dockerfile
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the container
COPY . .

# Allow non-root users to access files
RUN chmod -R 777 /app

# Expose the application port
EXPOSE 5000

# Set environment variable for the endpoint
ENV ENDPOINT_URL="http://default-endpoint"

# Use a non-root user to run the application
USER 1001

# Command to run the application
CMD ["python", "app.py"]
