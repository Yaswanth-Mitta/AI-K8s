# AI-K8s - Kubernetes NLP Interface

A natural language interface for managing Kubernetes clusters using AI.

## Features

- **Natural Language Commands**: Ask questions like "show me all pods" or "scale deployment frontend to 3 replicas"
- **Pod Diagnosis**: Get AI-powered analysis of failing pods with "diagnose pod [name]"
- **Real-time Chat Interface**: Web-based chat UI for easy interaction
- **Secure**: Input validation, XSS protection, and proper error handling
- **Kubernetes Integration**: Full RBAC support with minimal required permissions

## Quick Start

1. **Setup Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials
   ```

2. **Run Locally**:
   ```bash
   ./run_local.sh
   ```

3. **Deploy to Kubernetes**:
   ```bash
   ./build_and_push.sh  # Build and push Docker image
   ./deploy.sh          # Deploy to cluster
   ```

## Example Commands

- "show pods" - List all pods
- "list services" - Show all services  
- "get logs for pod-name" - View pod logs
- "scale deployment-name to 5" - Scale deployment
- "diagnose pod-name" - AI analysis of pod issues

## Architecture

- **FastAPI** backend with LangGraph workflow
- **AWS Bedrock** for natural language processing
- **Kubernetes Python Client** for cluster operations
- **Session management** with TTL cache
- **Security**: Input sanitization, XSS protection, resource limits

## Requirements

- Python 3.9+
- Kubernetes cluster access
- AWS Bedrock access
- Docker (for containerized deployment)