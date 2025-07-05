# Friday AI Trading System - MCP Server Guide

## Overview

Model Context Protocol (MCP) servers are a core component of the Friday AI Trading System, providing AI-powered capabilities for trading decisions. This guide explains how MCP servers work, how to configure them, and how to extend the system with custom MCP servers.

## Table of Contents

1. [Introduction to MCP Servers](#introduction-to-mcp-servers)
2. [Built-in MCP Servers](#built-in-mcp-servers)
3. [MCP Server Architecture](#mcp-server-architecture)
4. [Configuring MCP Servers](#configuring-mcp-servers)
5. [Creating Custom MCP Servers](#creating-custom-mcp-servers)
6. [Deploying MCP Servers](#deploying-mcp-servers)
7. [Testing MCP Servers](#testing-mcp-servers)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Introduction to MCP Servers

Model Context Protocol (MCP) servers are specialized AI services that enhance the Friday AI Trading System with advanced capabilities. They act as intelligent agents that can:

- Maintain memory of market conditions and trading history
- Perform sequential thinking for complex decision-making
- Extract knowledge from various data sources
- Generate trading signals based on market analysis
- Optimize portfolio allocation based on risk and return objectives

MCP servers communicate with the main system through a well-defined API, making them modular and extensible. This allows developers to create custom MCP servers to add new capabilities to the system.

## Built-in MCP Servers

The Friday AI Trading System comes with several built-in MCP servers:

### Memory MCP Server

The Memory MCP Server maintains context and remembers important trading information. It provides:

- Long-term memory of market conditions and trading history
- Short-term memory for recent market events
- Working memory for current trading decisions
- Episodic memory for specific market scenarios

### Sequential Thinking MCP Server

The Sequential Thinking MCP Server enables step-by-step reasoning for complex decisions. It provides:

- Chain-of-thought reasoning for trading decisions
- Multi-step analysis of market conditions
- Scenario planning and risk assessment
- Decision trees for trading strategies

### Knowledge Extraction MCP Server

The Knowledge Extraction MCP Server extracts insights from various data sources. It provides:

- Natural language processing for news and social media
- Sentiment analysis for market sentiment
- Entity recognition for relevant market actors
- Relationship extraction for market dynamics

## MCP Server Architecture

MCP servers follow a client-server architecture:

```
+----------------+      +----------------+
|                |      |                |
|  Friday Core   |<---->|   MCP Server   |
|   (Client)     |      |                |
|                |      |                |
+----------------+      +----------------+
```

The communication between the Friday Core and MCP servers is handled through a RESTful API or WebSocket connection, depending on the use case.

### MCP Server Components

A typical MCP server consists of the following components:

1. **API Layer**: Handles communication with the Friday Core
2. **Model Layer**: Implements the AI models and algorithms
3. **Memory Layer**: Manages state and persistence
4. **Integration Layer**: Connects with external services and data sources

### Communication Protocol

MCP servers communicate with the Friday Core using a JSON-based protocol. A typical request/response flow looks like this:

**Request:**

```json
{
  "action": "process",
  "data": {
    "context": {
      "market_data": { ... },
      "portfolio": { ... },
      "trading_history": [ ... ]
    },
    "query": "Should I buy AAPL based on current market conditions?"
  }
}
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "response": "Based on the current market conditions, buying AAPL is recommended. The stock has shown strong momentum over the past week, with increasing volume and positive technical indicators. The company's recent earnings report exceeded expectations, and the overall market sentiment is positive.",
    "confidence": 0.85,
    "reasoning": [
      "AAPL has shown strong upward momentum over the past week",
      "Volume has been increasing, indicating strong buying interest",
      "Technical indicators (RSI, MACD) are positive",
      "Recent earnings report exceeded expectations",
      "Overall market sentiment is positive"
    ],
    "recommendation": {
      "action": "buy",
      "symbol": "AAPL",
      "confidence": 0.85,
      "time_horizon": "medium_term"
    }
  }
}
```

## Configuring MCP Servers

MCP servers are configured in the `unified_config.py` file. The configuration includes:

- Server endpoints
- Authentication credentials
- Model parameters
- Memory settings
- Integration settings

Here's an example configuration for the built-in MCP servers:

```python
# MCP Server Configuration
MCP_SERVER_CONFIG = {
    "memory": {
        "enabled": True,
        "host": "localhost",
        "port": 8001,
        "api_key": os.getenv("MEMORY_MCP_API_KEY", ""),
        "memory_size": 1000,
        "ttl": 86400,  # 24 hours
        "persistence": {
            "enabled": True,
            "storage_path": "data/memory_mcp"
        }
    },
    "sequential_thinking": {
        "enabled": True,
        "host": "localhost",
        "port": 8002,
        "api_key": os.getenv("SEQUENTIAL_THINKING_MCP_API_KEY", ""),
        "max_steps": 10,
        "timeout": 30,  # seconds
        "model": {
            "name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    },
    "knowledge_extraction": {
        "enabled": True,
        "host": "localhost",
        "port": 8003,
        "api_key": os.getenv("KNOWLEDGE_EXTRACTION_MCP_API_KEY", ""),
        "sources": [
            "news",
            "social_media",
            "financial_reports",
            "economic_data"
        ],
        "update_frequency": 3600,  # 1 hour
        "max_age": 604800  # 7 days
    }
}
```

## Creating Custom MCP Servers

You can extend the Friday AI Trading System by creating custom MCP servers. This section provides a step-by-step guide for creating a custom MCP server.

### Step 1: Define the Server's Purpose

First, define the purpose and capabilities of your custom MCP server. For example, you might want to create a "Market Sentiment MCP Server" that analyzes news and social media to determine market sentiment.

### Step 2: Set Up the Project Structure

Create a new directory for your MCP server with the following structure:

```
market_sentiment_mcp/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── sentiment_model.py
│   ├── routers/
│   │   ├── __init__.py
│   │   └── sentiment_router.py
│   └── services/
│       ├── __init__.py
│       └── sentiment_service.py
├── data/
├── tests/
│   ├── __init__.py
│   └── test_sentiment.py
├── .env.example
├── Dockerfile
├── requirements.txt
└── README.md
```

### Step 3: Implement the API

Implement the API endpoints for your MCP server. Here's an example using FastAPI:

```python
# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.routers import sentiment_router
import uvicorn

app = FastAPI(
    title="Market Sentiment MCP Server",
    description="MCP Server for market sentiment analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sentiment_router.router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8004, reload=True)
```

```python
# app/routers/sentiment_router.py
from fastapi import APIRouter, HTTPException, Depends, Body
from app.services.sentiment_service import SentimentService
from typing import Dict, Any

router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])
sentiment_service = SentimentService()

@router.post("/analyze")
async def analyze_sentiment(data: Dict[str, Any] = Body(...)):
    try:
        result = sentiment_service.analyze_sentiment(data)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch_analyze")
async def batch_analyze_sentiment(data: Dict[str, Any] = Body(...)):
    try:
        result = sentiment_service.batch_analyze_sentiment(data)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 4: Implement the Service Layer

Implement the service layer that handles the business logic:

```python
# app/services/sentiment_service.py
from app.models.sentiment_model import SentimentModel
from typing import Dict, Any, List

class SentimentService:
    def __init__(self):
        self.model = SentimentModel()
    
    def analyze_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract text from data
        text = data.get("text", "")
        if not text:
            raise ValueError("Text is required for sentiment analysis")
        
        # Analyze sentiment
        sentiment = self.model.predict(text)
        
        # Return result
        return {
            "text": text,
            "sentiment": sentiment["sentiment"],
            "confidence": sentiment["confidence"],
            "entities": sentiment["entities"]
        }
    
    def batch_analyze_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract texts from data
        texts = data.get("texts", [])
        if not texts:
            raise ValueError("Texts are required for batch sentiment analysis")
        
        # Analyze sentiments
        results = []
        for text in texts:
            sentiment = self.model.predict(text)
            results.append({
                "text": text,
                "sentiment": sentiment["sentiment"],
                "confidence": sentiment["confidence"],
                "entities": sentiment["entities"]
            })
        
        # Return results
        return {
            "count": len(results),
            "results": results
        }
```

### Step 5: Implement the Model

Implement the model that performs the sentiment analysis:

```python
# app/models/sentiment_model.py
from typing import Dict, Any, List
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

class SentimentModel:
    def __init__(self):
        # Download NLTK resources
        nltk.download('vader_lexicon')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
    
    def predict(self, text: str) -> Dict[str, Any]:
        # Analyze sentiment
        sentiment_scores = self.sia.polarity_scores(text)
        
        # Determine sentiment
        if sentiment_scores["compound"] >= 0.05:
            sentiment = "positive"
        elif sentiment_scores["compound"] <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Return result
        return {
            "sentiment": sentiment,
            "confidence": abs(sentiment_scores["compound"]),
            "scores": sentiment_scores,
            "entities": entities
        }
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Tag parts of speech
        pos_tags = pos_tag(tokens)
        
        # Extract named entities
        named_entities = ne_chunk(pos_tags)
        
        # Process named entities
        entities = []
        for entity in named_entities:
            if hasattr(entity, 'label'):
                entity_text = ' '.join([child[0] for child in entity])
                entity_type = entity.label()
                entities.append({
                    "text": entity_text,
                    "type": entity_type
                })
        
        return entities
```

### Step 6: Create Requirements File

Create a `requirements.txt` file with the dependencies:

```
fastapi==0.95.1
uvicorn==0.22.0
nltk==3.8.1
pydantic==1.10.7
python-dotenv==1.0.0
requests==2.30.0
```

### Step 7: Create Dockerfile

Create a `Dockerfile` for containerization:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
```

### Step 8: Update the Friday Configuration

Update the `unified_config.py` file to include your custom MCP server:

```python
# MCP Server Configuration
MCP_SERVER_CONFIG = {
    # ... existing MCP servers ...
    "market_sentiment": {
        "enabled": True,
        "host": "localhost",
        "port": 8004,
        "api_key": os.getenv("MARKET_SENTIMENT_MCP_API_KEY", ""),
        "update_frequency": 3600,  # 1 hour
        "sources": [
            "news",
            "social_media",
            "financial_reports"
        ]
    }
}
```

## Deploying MCP Servers

MCP servers can be deployed in various ways, depending on your infrastructure and requirements.

### Local Deployment

For local development and testing, you can run MCP servers directly:

```bash
cd market_sentiment_mcp
python -m app.main
```

### Docker Deployment

For containerized deployment, you can use Docker:

```bash
cd market_sentiment_mcp
docker build -t market-sentiment-mcp .
docker run -p 8004:8004 market-sentiment-mcp
```

### Docker Compose

For multi-container deployment, you can use Docker Compose. Create a `docker-compose.yml` file:

```yaml
version: '3'

services:
  memory-mcp:
    build: ./memory_mcp
    ports:
      - "8001:8001"
    environment:
      - MEMORY_MCP_API_KEY=${MEMORY_MCP_API_KEY}
    volumes:
      - ./data/memory_mcp:/app/data

  sequential-thinking-mcp:
    build: ./sequential_thinking_mcp
    ports:
      - "8002:8002"
    environment:
      - SEQUENTIAL_THINKING_MCP_API_KEY=${SEQUENTIAL_THINKING_MCP_API_KEY}

  knowledge-extraction-mcp:
    build: ./knowledge_extraction_mcp
    ports:
      - "8003:8003"
    environment:
      - KNOWLEDGE_EXTRACTION_MCP_API_KEY=${KNOWLEDGE_EXTRACTION_MCP_API_KEY}

  market-sentiment-mcp:
    build: ./market_sentiment_mcp
    ports:
      - "8004:8004"
    environment:
      - MARKET_SENTIMENT_MCP_API_KEY=${MARKET_SENTIMENT_MCP_API_KEY}
```

Then run:

```bash
docker-compose up -d
```

### Kubernetes Deployment

For production deployment, you can use Kubernetes. Create a `kubernetes` directory with the following files:

```yaml
# kubernetes/market-sentiment-mcp-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: market-sentiment-mcp
  labels:
    app: market-sentiment-mcp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: market-sentiment-mcp
  template:
    metadata:
      labels:
        app: market-sentiment-mcp
    spec:
      containers:
      - name: market-sentiment-mcp
        image: market-sentiment-mcp:latest
        ports:
        - containerPort: 8004
        env:
        - name: MARKET_SENTIMENT_MCP_API_KEY
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: market-sentiment-mcp-api-key
```

```yaml
# kubernetes/market-sentiment-mcp-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: market-sentiment-mcp
spec:
  selector:
    app: market-sentiment-mcp
  ports:
  - port: 8004
    targetPort: 8004
  type: ClusterIP
```

```yaml
# kubernetes/mcp-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: mcp-secrets
type: Opaque
data:
  memory-mcp-api-key: <base64-encoded-key>
  sequential-thinking-mcp-api-key: <base64-encoded-key>
  knowledge-extraction-mcp-api-key: <base64-encoded-key>
  market-sentiment-mcp-api-key: <base64-encoded-key>
```

Then apply the Kubernetes manifests:

```bash
kubectl apply -f kubernetes/
```

## Testing MCP Servers

Testing is crucial for ensuring the reliability and correctness of your MCP servers. This section provides guidance on testing MCP servers.

### Unit Testing

Create unit tests for your MCP server components:

```python
# tests/test_sentiment.py
import unittest
from app.models.sentiment_model import SentimentModel
from app.services.sentiment_service import SentimentService

class TestSentimentModel(unittest.TestCase):
    def setUp(self):
        self.model = SentimentModel()
    
    def test_positive_sentiment(self):
        text = "The company reported excellent earnings, exceeding expectations."
        result = self.model.predict(text)
        self.assertEqual(result["sentiment"], "positive")
        self.assertGreaterEqual(result["confidence"], 0.05)
    
    def test_negative_sentiment(self):
        text = "The stock crashed after disappointing quarterly results."
        result = self.model.predict(text)
        self.assertEqual(result["sentiment"], "negative")
        self.assertGreaterEqual(result["confidence"], 0.05)
    
    def test_neutral_sentiment(self):
        text = "The company released its quarterly report today."
        result = self.model.predict(text)
        self.assertEqual(result["sentiment"], "neutral")
        self.assertLess(result["confidence"], 0.05)

class TestSentimentService(unittest.TestCase):
    def setUp(self):
        self.service = SentimentService()
    
    def test_analyze_sentiment(self):
        data = {"text": "The company reported excellent earnings, exceeding expectations."}
        result = self.service.analyze_sentiment(data)
        self.assertEqual(result["sentiment"], "positive")
        self.assertGreaterEqual(result["confidence"], 0.05)
        self.assertEqual(result["text"], data["text"])
    
    def test_batch_analyze_sentiment(self):
        data = {
            "texts": [
                "The company reported excellent earnings, exceeding expectations.",
                "The stock crashed after disappointing quarterly results."
            ]
        }
        result = self.service.batch_analyze_sentiment(data)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["results"][0]["sentiment"], "positive")
        self.assertEqual(result["results"][1]["sentiment"], "negative")

if __name__ == "__main__":
    unittest.main()
```

### Integration Testing

Create integration tests to verify the API endpoints:

```python
# tests/test_integration.py
import unittest
import requests

class TestSentimentAPI(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:8004/api/v1/sentiment"
    
    def test_analyze_sentiment(self):
        data = {"text": "The company reported excellent earnings, exceeding expectations."}
        response = requests.post(f"{self.base_url}/analyze", json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["sentiment"], "positive")
    
    def test_batch_analyze_sentiment(self):
        data = {
            "texts": [
                "The company reported excellent earnings, exceeding expectations.",
                "The stock crashed after disappointing quarterly results."
            ]
        }
        response = requests.post(f"{self.base_url}/batch_analyze", json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["count"], 2)
        self.assertEqual(result["data"]["results"][0]["sentiment"], "positive")
        self.assertEqual(result["data"]["results"][1]["sentiment"], "negative")

if __name__ == "__main__":
    unittest.main()
```

### Load Testing

Perform load testing to ensure your MCP server can handle the expected load:

```python
# tests/test_load.py
import time
import concurrent.futures
import requests
import statistics

def send_request(url, data):
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()
    return {
        "status_code": response.status_code,
        "response_time": end_time - start_time
    }

def run_load_test(url, data, num_requests, concurrency):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, url, data) for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # Calculate statistics
    response_times = [result["response_time"] for result in results]
    success_count = sum(1 for result in results if result["status_code"] == 200)
    
    print(f"Total requests: {num_requests}")
    print(f"Successful requests: {success_count}")
    print(f"Success rate: {success_count / num_requests * 100:.2f}%")
    print(f"Min response time: {min(response_times):.4f} seconds")
    print(f"Max response time: {max(response_times):.4f} seconds")
    print(f"Average response time: {statistics.mean(response_times):.4f} seconds")
    print(f"Median response time: {statistics.median(response_times):.4f} seconds")
    print(f"95th percentile response time: {sorted(response_times)[int(num_requests * 0.95)]:.4f} seconds")

if __name__ == "__main__":
    url = "http://localhost:8004/api/v1/sentiment/analyze"
    data = {"text": "The company reported excellent earnings, exceeding expectations."}
    num_requests = 1000
    concurrency = 10
    
    run_load_test(url, data, num_requests, concurrency)
```

## Best Practices

Follow these best practices when developing and deploying MCP servers:

### Development Best Practices

1. **Modular Design**: Design your MCP server with modularity in mind, separating concerns into different components.

2. **Error Handling**: Implement robust error handling to gracefully handle failures and provide meaningful error messages.

3. **Logging**: Implement comprehensive logging to facilitate debugging and monitoring.

4. **Documentation**: Document your code, API endpoints, and configuration options.

5. **Testing**: Write unit tests, integration tests, and load tests to ensure reliability and performance.

### Deployment Best Practices

1. **Containerization**: Use Docker to containerize your MCP servers for consistent deployment across environments.

2. **Orchestration**: Use Kubernetes or Docker Compose for orchestrating multiple MCP servers.

3. **Scaling**: Design your MCP servers to scale horizontally to handle increased load.

4. **Monitoring**: Implement monitoring and alerting to detect and respond to issues.

5. **Security**: Secure your MCP servers with API keys, TLS encryption, and proper access controls.

### Performance Best Practices

1. **Caching**: Implement caching to reduce computation and improve response times.

2. **Asynchronous Processing**: Use asynchronous processing for I/O-bound operations.

3. **Batch Processing**: Implement batch processing for handling multiple requests efficiently.

4. **Resource Management**: Properly manage resources like memory and CPU to prevent bottlenecks.

5. **Optimization**: Profile your code and optimize performance-critical sections.

## Troubleshooting

This section provides guidance on troubleshooting common issues with MCP servers.

### Connection Issues

**Symptom**: The Friday Core cannot connect to the MCP server.

**Possible Causes**:
- The MCP server is not running
- The MCP server is running on a different host or port than configured
- Firewall or network issues are blocking the connection

**Solutions**:
- Verify that the MCP server is running: `ps aux | grep mcp`
- Check the MCP server logs for errors
- Verify the host and port configuration in `unified_config.py`
- Check firewall rules and network connectivity

### Authentication Issues

**Symptom**: The Friday Core receives authentication errors when connecting to the MCP server.

**Possible Causes**:
- The API key is incorrect or missing
- The API key has expired
- The MCP server's authentication mechanism has changed

**Solutions**:
- Verify the API key in `unified_config.py`
- Check the MCP server logs for authentication errors
- Regenerate the API key if necessary

### Performance Issues

**Symptom**: The MCP server is slow to respond or becomes unresponsive under load.

**Possible Causes**:
- The MCP server is overloaded
- The MCP server is running out of resources
- The MCP server has a memory leak

**Solutions**:
- Scale the MCP server horizontally by adding more instances
- Increase the resources (CPU, memory) allocated to the MCP server
- Implement caching to reduce computation
- Profile the MCP server to identify and fix performance bottlenecks
- Check for memory leaks using tools like `memory_profiler`

### Model Issues

**Symptom**: The MCP server returns incorrect or unexpected results.

**Possible Causes**:
- The model is not properly trained or configured
- The input data is in an unexpected format
- The model has a bug or limitation

**Solutions**:
- Verify the model configuration
- Check the input data format
- Test the model with known inputs and expected outputs
- Retrain the model if necessary
- Update the model implementation to fix bugs or limitations

---

For more information, refer to the `README.md` and other documentation files in the project root directory.