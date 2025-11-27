#!/bin/bash
set -e

echo "=================================================="
echo "Installing Kubeflow Pipelines on kind cluster"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
CLUSTER_NAME="mlops-workshop"
PIPELINE_VERSION="2.14.3"  # Compatible with kfp SDK 2.9.0

# Check prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl found${NC}"

if ! command -v kind &> /dev/null; then
    echo -e "${RED}Error: kind not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kind found${NC}"

# Check if cluster exists, create if not
if ! kind get clusters | grep -q "$CLUSTER_NAME"; then
    echo -e "${YELLOW}kind cluster '$CLUSTER_NAME' not found. Creating it...${NC}"
    kind create cluster --name "$CLUSTER_NAME"
    echo -e "${GREEN}✓ kind cluster created${NC}"
else
    echo -e "${GREEN}✓ kind cluster '$CLUSTER_NAME' found${NC}"
fi
echo ""

# Install Kubeflow Pipelines
echo -e "${YELLOW}Step 2: Installing Kubeflow Pipelines (Standalone)${NC}"
echo "Version: $PIPELINE_VERSION"
echo "This may take several minutes..."
echo ""

echo -e "${YELLOW}Installing cluster-scoped resources...${NC}"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

echo -e "${YELLOW}Installing cert-manager resources...${NC}"
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.18.2/cert-manager.yaml
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s

echo -e "${YELLOW}Installing Kubeflow Pipelines components...${NC}"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/cert-manager/platform-agnostic-k8s-native?ref=$PIPELINE_VERSION"

echo -e "${GREEN}✓ Kubeflow Pipelines manifests applied${NC}"
echo ""

# Wait for pods to be ready
echo -e "${YELLOW}Step 3: Waiting for pods to start...${NC}"
echo "This may take 3-5 minutes for images to download and pods to start..."
echo ""

# Wait for kubeflow namespace to exist
echo -e "${YELLOW}Waiting for kubeflow namespace...${NC}"
for i in {1..30}; do
    if kubectl get namespace kubeflow &> /dev/null; then
        echo -e "${GREEN}✓ kubeflow namespace ready${NC}"
        break
    fi
    sleep 2
done

# Fix minio image (common issue with default minio image)
echo ""
echo -e "${YELLOW}Patching minio deployment with compatible image...${NC}"
for i in {1..30}; do
    if kubectl get deployment minio -n kubeflow &> /dev/null; then
        kubectl set image deployment/minio -n kubeflow minio=minio/minio:RELEASE.2025-09-07T16-13-09Z-cpuv1
        echo -e "${GREEN}✓ minio deployment patched${NC}"
        break
    fi
    echo "Waiting for minio deployment to be created..."
    sleep 2
done

# Give pods time to start
sleep 10

# Show pod status
echo ""
echo -e "${YELLOW}Current pod status in kubeflow namespace:${NC}"
kubectl get pods -n kubeflow
echo ""

# Wait for critical pods (with timeout and better error handling)
echo -e "${YELLOW}Waiting for pods to become ready (this may take a few minutes)...${NC}"
echo "You can press Ctrl+C and run this manually later:"
echo "  kubectl wait --for=condition=Ready --timeout=300s -n kubeflow pod --all"
echo ""

# Wait with error handling
kubectl wait --for=condition=Ready --timeout=600s -n kubeflow pod --all 2>/dev/null || {
    echo -e "${YELLOW}⚠ Some pods are not ready yet. This is normal for first-time installation.${NC}"
    echo -e "${YELLOW}Check status with: kubectl get pods -n kubeflow${NC}"
}

echo ""
echo -e "${GREEN}=================================================="
echo "Installation Complete!"
echo "==================================================${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo ""
echo "1. Check pod status (wait until all pods are Running):"
echo "   kubectl get pods -n kubeflow"
echo ""
echo "2. Watch pods until all are ready (Ctrl+C to exit):"
echo "   kubectl get pods -n kubeflow -w"
echo ""
echo "3. Once ready, port-forward to access UI:"
echo "   kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
echo ""
echo "4. Open browser:"
echo "   http://localhost:8080"
echo ""
echo -e "${YELLOW}Note: First-time installation downloads large images.${NC}"
echo -e "${YELLOW}It may take 5-10 minutes for all pods to be ready.${NC}"
echo ""
