FROM python:3.12-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models during build (no internet needed at runtime)
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
RUN python -c "from lightglue import LightGlue, SuperPoint; SuperPoint(max_num_keypoints=2048).eval(); LightGlue(features='superpoint').eval()"

# Application code
COPY . .

# Create storage directories
RUN mkdir -p storage/baselines storage/heatmaps ml_models

# Non-root user
RUN useradd -m -r pcd && chown -R pcd:pcd /app
USER pcd

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
