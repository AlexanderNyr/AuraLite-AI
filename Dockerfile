FROM python:3.10-slim

# Install system dependencies for tkinter and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Default command: launch the GUI
CMD ["python", "gui_app.py"]
