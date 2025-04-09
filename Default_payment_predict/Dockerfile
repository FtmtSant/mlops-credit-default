FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt &&\
    pip install streamlit && streamlit --version\
    pip uninstall -y lightgbm && pip install --no-cache-dir lightgbm

# Expose Streamlit port
EXPOSE 8501


# Set environment variables for Arize AI (these values are passed at build time)
ARG ARIZE_API_KEY
ARG ARIZE_SPACE_KEY
ENV ARIZE_API_KEY=$ARIZE_API_KEY
ENV ARIZE_SPACE_KEY=$ARIZE_SPACE_KEY

# Set entrypoint and command
ENTRYPOINT ["streamlit"]
CMD ["run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
