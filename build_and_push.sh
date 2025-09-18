#!/bin/bash
# This script automates building and pushing the Docker image.

set -e

IMAGE_NAME="ai-k8s-chat"
IMAGE_TAG="latest"

echo "🚀 Starting build and push process..."

# --- Step 1: Check for and load .env file ---
echo "🔎 Checking for .env file..."
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found. Please create one from .env.example and fill it out."
    exit 1
fi

# Source the .env file in a way that handles special characters safely.
set -a
source .env
set +a

echo "✅ .env file loaded."

# --- Step 2: Verify required DOCKER_USERNAME ---
echo "🔎 Verifying Docker Hub username..."
if [ -z "${DOCKER_USERNAME}" ]; then
    echo "❌ Error: DOCKER_USERNAME is not set in the .env file."
    exit 1
fi
echo "✅ Docker Hub username '${DOCKER_USERNAME}' found."


FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo
echo "Building image: ${FULL_IMAGE_NAME}"
echo

docker build -t "${FULL_IMAGE_NAME}" . || { echo "Docker build failed"; exit 1; }

echo
echo "Logging into Docker Hub..."
docker login || { echo "Docker login failed"; exit 1; }

echo
echo "Pushing image to Docker Hub..."
docker push "${FULL_IMAGE_NAME}" || { echo "Docker push failed"; exit 1; }

echo
echo "--- Success! ---"
echo
echo "Image ${FULL_IMAGE_NAME} has been pushed."