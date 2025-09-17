#!/bin/bash
# This script automates the deployment of the AI K8s Chat Manager.

set -e

echo "🚀 Starting deployment process..."

# --- Step 1: Check for and load .env file ---
echo "🔎 Checking for .env file..."
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found. Please create one from .env.example and fill in your AWS credentials."
    exit 1
fi

# Export variables from .env file for use in this script
export $(grep -v '^#' .env | xargs)

echo "✅ .env file loaded."

# --- Step 2: Verify required variables ---
echo "🔎 Verifying credentials..."
if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    echo "❌ Error: AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not set in the .env file."
    exit 1
fi
echo "✅ Credentials found."

# --- Step 3: Create/Update Kubernetes Secret ---
echo "🔐 Creating/updating Kubernetes secret 'bedrock-credentials' நான...

# This command is idempotent. It creates the secret if it doesn't exist, and updates it if it does.
kubectl create secret generic bedrock-credentials \
  --from-literal=aws_access_key_id="${AWS_ACCESS_KEY_ID}" \
  --from-literal=aws_secret_access_key="${AWS_SECRET_ACCESS_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Secret 'bedrock-credentials' is up to date."

# --- Step 4: Apply all Kubernetes manifests ---
echo "Applying all Kubernetes manifests from the 'k8s' directory..."

# The -k flag for kustomize is also a good option here, but -f is more direct for this project structure.
kubectl apply -f k8s/

echo "✅ All manifests applied."

# --- Step 5: Final Instructions ---
echo ""
echo "🎉 Deployment complete!"

echo ""
echo "To check the status of your application, run:"
echo "  kubectl get pods -l app=ai-k8s-chat"

echo "To find the port and access the UI, run:"
echo "  kubectl get service ai-k8s-chat-service"
