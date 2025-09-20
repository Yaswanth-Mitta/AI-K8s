#!/bin/bash
# This script automates the deployment of the AI K8s Chat Manager.

set -e

echo "🚀 Starting deployment process..."

# --- Step 0: Install NGINX Ingress Controller if not present ---
echo "🔎 Checking NGINX Ingress Controller..."
if ! kubectl get pods -n ingress-nginx | grep -q "ingress-nginx-controller.*Running"; then
    echo "📦 Installing NGINX Ingress Controller..."
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/baremetal/deploy.yaml
    echo "⏳ Waiting for ingress controller to be ready..."
    kubectl wait --namespace ingress-nginx --for=condition=ready pod --selector=app.kubernetes.io/component=controller --timeout=300s
else
    echo "✅ NGINX Ingress Controller already running."
fi

# --- Step 1: Check for and load .env file ---
echo "🔎 Checking for .env file..."
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found. Please create one from .env.example and fill in your AWS credentials."
    exit 1
fi

# Source the .env file in a way that handles special characters safely.
set -a
source .env
set +a

echo "✅ .env file loaded."

# --- Step 2: Verify required variables ---
echo "🔎 Verifying credentials..."
if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    echo "❌ Error: AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not set in the .env file."
    exit 1
fi
echo "✅ Credentials found."

# --- Step 3: Validate kubectl availability ---
if ! command -v kubectl &> /dev/null; then
    echo "❌ Error: kubectl is not installed or not in PATH"
    exit 1
fi

# --- Step 4: Create/Update Kubernetes Secret ---
echo "🔐 Creating/updating Kubernetes secret 'bedrock-credentials'..."

# This command is idempotent. It creates the secret if it doesn't exist, and updates it if it does.
kubectl create secret generic bedrock-credentials \
  --from-literal=aws_access_key_id="${AWS_ACCESS_KEY_ID}" \
  --from-literal=aws_secret_access_key="${AWS_SECRET_ACCESS_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f - || { echo "Failed to create/update secret"; exit 1; }

echo "✅ Secret 'bedrock-credentials' is up to date."

# --- Step 5: Validate k8s directory ---
if [ ! -d "k8s" ]; then
    echo "❌ Error: k8s directory not found"
    exit 1
fi

# --- Step 6: Apply all Kubernetes manifests ---
echo "Applying all Kubernetes manifests from the 'k8s' directory..."

kubectl apply -f k8s/ || { echo "Failed to apply Kubernetes manifests"; exit 1; }

echo "✅ All manifests applied."

# --- Step 5: Final Instructions ---
echo ""
echo "🎉 Deployment complete!"

echo ""
echo "To check the status of your application, run:"
echo "  kubectl get pods -l app=ai-k8s-chat"

echo "To access the UI:"
echo "  # Get NodePort for ingress controller:"
echo "  kubectl get svc -n ingress-nginx ingress-nginx-controller"
echo "  # Access via: http://<EC2_PUBLIC_IP>:<NODEPORT>"
echo "  # Current NodePort should be: 31687"
echo "  # Make sure EC2 security group allows the NodePort (31687)"