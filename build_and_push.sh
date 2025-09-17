#!/bin/bash
set -e

IMAGE_NAME="ai-k8s-chat"
IMAGE_TAG="latest"

# Prompt for Docker Hub username
read -p "Please enter your Docker Hub username and press ENTER: " DOCKER_USERNAME

if [ -z "$DOCKER_USERNAME" ]; then
    echo "Username cannot be empty."
    exit 1
fi

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo
echo "Building image: ${FULL_IMAGE_NAME}"
echo

# Build, Push, and Login
docker build -t "${FULL_IMAGE_NAME}" .

echo
echo "Logging into Docker Hub. Please enter your credentials."
docker login

echo
echo "Pushing image to Docker Hub..."
docker push "${FULL_IMAGE_NAME}"

echo
echo "--- Success! ---"
echo
echo "Image ${FULL_IMAGE_NAME} has been pushed."
echo
echo "IMPORTANT: Remember to update the 'image:' line in k8s/deployment.yaml to match this image name before you deploy."
