# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only the necessary files into the container
COPY ./ ./

# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Healthcheck to verify container health
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set entry point for the container, running the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
