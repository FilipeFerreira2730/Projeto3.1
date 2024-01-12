FROM python:3.9.15-slim

WORKDIR /usr/src/app

# Install nodemon and other system dependencies
RUN apt-get update \
    && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash \
    && apt-get install -y nodejs \
    && npm install -g npm@9.2.0 \
    && npm install -g nodemon \
    && apt-get install -y libpq-dev build-essential

# Copy only the requirements file
COPY requirements.txt ./
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install Streamlit
RUN pip install streamlit

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run Streamlit app when the container launches
CMD ["streamlit", "run", "main.py"]
