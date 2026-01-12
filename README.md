# Autonomous Vision Agent

A comprehensive autonomous vision agent system designed to process, analyze, and respond to visual data with minimal human intervention. This intelligent system leverages cutting-edge computer vision and machine learning technologies to enable autonomous decision-making and task execution based on visual input.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Autonomous Vision Agent is a modular system that combines computer vision, machine learning, and autonomous decision-making capabilities. It can be deployed in various environments to perform real-time visual analysis, object detection, scene understanding, and autonomous task execution.

### Key Capabilities

- Real-time video stream processing
- Multi-object detection and classification
- Scene understanding and contextual analysis
- Autonomous decision-making based on visual input
- Configurable action execution pipelines
- Comprehensive logging and monitoring

## Features

- **Multi-Model Support**: Compatible with various pre-trained and custom vision models
- **Real-Time Processing**: Optimized for low-latency inference on edge and cloud devices
- **Modular Architecture**: Easily extensible with custom processors and handlers
- **Robust Error Handling**: Comprehensive error detection and recovery mechanisms
- **Scalable Deployment**: Support for single-instance and distributed deployments
- **Advanced Monitoring**: Built-in metrics, logging, and alerting capabilities
- **API-Driven**: RESTful API for remote control and monitoring
- **Configuration Management**: Flexible YAML-based configuration system

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **OS**: Linux (Ubuntu 18.04+), macOS 10.14+, or Windows 10+
- **RAM**: Minimum 8GB (16GB+ recommended for optimal performance)
- **GPU**: NVIDIA CUDA-compatible GPU (optional, for accelerated inference)
- **Disk Space**: 10GB+ for models and dependencies

### Prerequisites

Before installing the Autonomous Vision Agent, ensure you have the following installed:

```bash
# Update package manager
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    libopencv-dev \
    python3-opencv
```

### Step-by-Step Installation

1. **Clone the Repository**

```bash
git clone https://github.com/adrudanski/utonomous-vision-agent.git
cd utonomous-vision-agent
```

2. **Create Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

4. **Download Pre-trained Models** (Optional)

```bash
python scripts/download_models.py
```

5. **Verify Installation**

```bash
python -m autonomous_vision_agent --version
python -m pytest tests/ -v  # Run test suite
```

### GPU Support (Optional)

For NVIDIA GPU support:

```bash
# Install CUDA toolkit (if not already installed)
# Follow NVIDIA's official documentation for your OS

# Install GPU-accelerated dependencies
pip install -r requirements-gpu.txt
```

## Configuration

### Configuration File Structure

The agent uses YAML-based configuration. Create a `config.yaml` file in the project root:

```yaml
# config.yaml
agent:
  name: "autonomous-vision-agent"
  version: "1.0.0"
  debug: false

vision:
  model:
    type: "yolov8"  # Options: yolov8, yolov5, faster-rcnn, ssd
    weights: "models/yolov8n.pt"
    confidence_threshold: 0.5
    nms_threshold: 0.45
  
  input:
    source: "camera"  # Options: camera, rtsp, file, image
    camera_id: 0
    fps: 30
    resolution: [640, 480]
  
  processing:
    enable_gpu: true
    batch_size: 1
    num_workers: 4

decision_engine:
  enabled: true
  rules_file: "config/decision_rules.yaml"
  timeout: 5.0

actions:
  enabled: true
  handlers:
    - type: "webhook"
      url: "http://localhost:8080/api/actions"
      timeout: 10
    - type: "mqtt"
      broker: "mqtt.example.com"
      port: 1883
      topic: "autonomous-agent/actions"

logging:
  level: "INFO"
  file: "logs/agent.log"
  max_size: "100MB"
  backup_count: 5
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30
```

### Environment Variables

```bash
# .env file
AVA_CONFIG_PATH="./config.yaml"
AVA_LOG_LEVEL="INFO"
AVA_ENABLE_GPU="true"
AVA_API_PORT="8000"
AVA_API_HOST="0.0.0.0"
```

## Deployment

### Local Deployment

1. **Start the Agent**

```bash
python -m autonomous_vision_agent --config config.yaml
```

2. **Access the Web Interface**

Open your browser and navigate to `http://localhost:8000`

### Docker Deployment

1. **Build Docker Image**

```bash
docker build -t autonomous-vision-agent:latest .
```

2. **Run Docker Container**

```bash
docker run -d \
  --name ava-instance \
  --gpus all \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/logs:/app/logs \
  autonomous-vision-agent:latest
```

3. **View Logs**

```bash
docker logs -f ava-instance
```

### Kubernetes Deployment

1. **Create ConfigMap**

```bash
kubectl create configmap ava-config --from-file=config.yaml
```

2. **Deploy Using Helm**

```bash
helm install ava ./helm/autonomous-vision-agent \
  --namespace vision-agents \
  --values values.yaml
```

3. **Verify Deployment**

```bash
kubectl get pods -n vision-agents
kubectl logs -f deployment/ava -n vision-agents
```

### Cloud Deployment

#### AWS (SageMaker)

```bash
# Package the application
python scripts/prepare_sagemaker.py

# Deploy to SageMaker
aws sagemaker create-model \
  --model-name ava-model \
  --primary-container Image=<ECR_URI>,ModelDataUrl=s3://<BUCKET>/model.tar.gz
```

#### Azure (Container Instances)

```bash
# Create container group
az container create \
  --resource-group myResourceGroup \
  --name ava-instance \
  --image <ACR_URI>/autonomous-vision-agent:latest \
  --ports 8000 9090 \
  --environment-variables AVA_LOG_LEVEL=INFO
```

#### Google Cloud (Run)

```bash
gcloud run deploy ava \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --timeout 3600
```

## Usage

### Basic Usage

```python
from autonomous_vision_agent import AutonomousVisionAgent

# Initialize the agent
agent = AutonomousVisionAgent(config_path="config.yaml")

# Start processing
agent.start()

# Process a single image
results = agent.process_image("path/to/image.jpg")
print(results)

# Stop the agent
agent.stop()
```

### Real-Time Video Processing

```python
from autonomous_vision_agent import AutonomousVisionAgent

agent = AutonomousVisionAgent(config_path="config.yaml")
agent.start()

# Process video stream (camera)
agent.process_stream(source="camera", camera_id=0)

# Process RTSP stream
agent.process_stream(source="rtsp://example.com/stream")

# Graceful shutdown
agent.stop()
```

### REST API Usage

```bash
# Start the API server
python -m autonomous_vision_agent --api --port 8000

# Process an image via API
curl -X POST http://localhost:8000/api/process \
  -F "image=@test_image.jpg"

# Get agent status
curl http://localhost:8000/api/status

# Get performance metrics
curl http://localhost:8000/api/metrics
```

### Advanced Configuration

```python
from autonomous_vision_agent import AutonomousVisionAgent
from autonomous_vision_agent.config import Config

# Load and customize configuration
config = Config.from_file("config.yaml")
config.vision.model.confidence_threshold = 0.7
config.vision.input.fps = 60

# Initialize with custom config
agent = AutonomousVisionAgent(config=config)
agent.start()
```

## API Reference

### Endpoints

#### POST /api/process

Process an image and return detection results.

**Request:**
```bash
curl -X POST http://localhost:8000/api/process \
  -F "image=@image.jpg" \
  -F "return_annotated=true"
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 150, 250, 400],
      "action": "log_detection"
    }
  ],
  "processing_time_ms": 45,
  "timestamp": "2026-01-12T18:39:30Z"
}
```

#### GET /api/status

Get current agent status and statistics.

**Response:**
```json
{
  "status": "running",
  "uptime_seconds": 3600,
  "processed_frames": 1800,
  "average_inference_time_ms": 50,
  "active_detections": 5,
  "gpu_memory_usage": "2.5GB / 8.0GB",
  "timestamp": "2026-01-12T18:39:30Z"
}
```

#### GET /api/metrics

Get detailed performance metrics.

**Response:**
```json
{
  "metrics": {
    "fps": 30,
    "inference_time_mean": 50,
    "inference_time_std": 5,
    "detection_rate": 0.85,
    "false_positive_rate": 0.02
  }
}
```

#### POST /api/config

Update configuration at runtime.

**Request:**
```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "vision.model.confidence_threshold": 0.6
  }'
```

#### POST /api/shutdown

Gracefully shutdown the agent.

```bash
curl -X POST http://localhost:8000/api/shutdown
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Out of Memory (OOM) Error

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
# In config.yaml, set: vision.processing.batch_size = 1

# Reduce model size
# Use YOLOv8n (nano) instead of YOLOv8m (medium)

# Reduce input resolution
# Set: vision.input.resolution = [320, 240]

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### Issue: Low Frame Rate

**Symptom**: Processing speed < expected FPS

**Solutions**:
```bash
# Enable GPU acceleration
# In config.yaml, set: vision.processing.enable_gpu = true

# Increase num_workers
# Set: vision.processing.num_workers = 8

# Reduce model complexity
# Use smaller model: yolov8n instead of yolov8l

# Optimize resolution
# Lower resolution = faster processing
# Set: vision.input.resolution = [480, 360]
```

#### Issue: No Detections

**Symptom**: Agent runs but returns no detections

**Solutions**:
```bash
# Adjust confidence threshold
# In config.yaml, lower: vision.model.confidence_threshold = 0.3

# Check model compatibility
python scripts/test_model.py --model-path models/yolov8n.pt

# Verify input source
python scripts/test_input.py --source camera

# Check lighting conditions
# Ensure adequate lighting for camera input
```

#### Issue: API Connection Timeout

**Symptom**: `ConnectionError: Unable to connect to localhost:8000`

**Solutions**:
```bash
# Check if agent is running
ps aux | grep autonomous_vision_agent

# Verify port is not in use
lsof -i :8000

# Check firewall rules
sudo ufw allow 8000/tcp

# Restart with verbose logging
python -m autonomous_vision_agent --log-level DEBUG
```

#### Issue: High CPU Usage

**Symptom**: CPU usage > 90%

**Solutions**:
```bash
# Reduce processing threads
# Set: vision.processing.num_workers = 2

# Lower FPS target
# Set: vision.input.fps = 15

# Enable GPU to offload CPU
# Set: vision.processing.enable_gpu = true

# Monitor resource usage
python scripts/monitor_resources.py
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Command line
python -m autonomous_vision_agent --debug

# Configuration file
agent:
  debug: true
  
# Environment variable
export AVA_LOG_LEVEL="DEBUG"
```

### Log Analysis

```bash
# View recent logs
tail -f logs/agent.log

# Filter errors
grep "ERROR" logs/agent.log | tail -20

# Search for specific patterns
grep "detection_failed\|timeout" logs/agent.log

# Generate log report
python scripts/analyze_logs.py --log-file logs/agent.log
```

### Performance Profiling

```bash
# Profile inference time
python scripts/profile_inference.py --config config.yaml

# Benchmark model
python scripts/benchmark_model.py --model yolov8n --iterations 100

# Memory profiling
python -m memory_profiler scripts/run_agent.py
```

### Getting Help

- **Documentation**: Check `/docs` folder for detailed guides
- **Issues**: Report bugs at https://github.com/adrudanski/utonomous-vision-agent/issues
- **Discussions**: Join community discussions at GitHub Discussions
- **Email Support**: contact@example.com

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/adrudanski/utonomous-vision-agent.git
cd utonomous-vision-agent
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting and formatting
flake8 autonomous_vision_agent/
black autonomous_vision_agent/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated**: 2026-01-12
**Version**: 1.0.0
**Maintainer**: adrudanski