#!/bin/bash

# Verification script for Kubeflow Pipelines installation

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=================================================="
echo "Verifying Kubeflow Pipelines Installation"
echo "=================================================="
echo ""

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl installed${NC}"

# Check kind cluster
if ! kind get clusters | grep -q "mlops-workshop"; then
    echo -e "${RED}✗ kind cluster 'mlops-workshop' not found${NC}"
    echo "  Run: kind create cluster --name mlops-workshop"
    exit 1
fi
echo -e "${GREEN}✓ kind cluster 'mlops-workshop' exists${NC}"

# Check kubeflow namespace
if ! kubectl get namespace kubeflow &> /dev/null; then
    echo -e "${RED}✗ kubeflow namespace not found${NC}"
    echo "  Run: ./scripts/install-kubeflow.sh"
    exit 1
fi
echo -e "${GREEN}✓ kubeflow namespace exists${NC}"

# Check pods
echo ""
echo -e "${YELLOW}Checking pod status...${NC}"
echo ""

POD_STATUS=$(kubectl get pods -n kubeflow --no-headers 2>/dev/null)
TOTAL_PODS=$(echo "$POD_STATUS" | wc -l | tr -d ' ')
RUNNING_PODS=$(echo "$POD_STATUS" | grep -c "Running" || echo "0")
READY_PODS=$(echo "$POD_STATUS" | grep -c "1/1\|2/2" || echo "0")

echo "Total pods: $TOTAL_PODS"
echo "Running pods: $RUNNING_PODS"
echo "Ready pods: $READY_PODS"
echo ""

kubectl get pods -n kubeflow

echo ""

if [ "$RUNNING_PODS" -eq "$TOTAL_PODS" ] && [ "$READY_PODS" -eq "$TOTAL_PODS" ]; then
    echo -e "${GREEN}✓ All pods are Running and Ready${NC}"
    PODS_OK=true
else
    echo -e "${YELLOW}⚠ Some pods are not ready yet${NC}"
    echo "  This is normal during first installation."
    echo "  Wait a few more minutes and run this script again."
    PODS_OK=false
fi

# Check services
echo ""
echo -e "${YELLOW}Checking services...${NC}"
if kubectl get svc -n kubeflow ml-pipeline-ui &> /dev/null; then
    echo -e "${GREEN}✓ ml-pipeline-ui service exists${NC}"
else
    echo -e "${RED}✗ ml-pipeline-ui service not found${NC}"
    exit 1
fi

# Check if port-forward is running
echo ""
echo -e "${YELLOW}Checking UI access...${NC}"
if lsof -i :8080 &> /dev/null; then
    echo -e "${GREEN}✓ Port 8080 is in use (port-forward may be running)${NC}"
    echo "  Access UI at: http://localhost:8080"
else
    echo -e "${YELLOW}⚠ Port 8080 not in use${NC}"
    echo "  To access UI, run in a separate terminal:"
    echo "  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
fi

# Final summary
echo ""
echo "=================================================="
if [ "$PODS_OK" = true ]; then
    echo -e "${GREEN}✓ Installation Verified Successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start port-forward (if not already running):"
    echo "   kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
    echo ""
    echo "2. Open browser: http://localhost:8080"
    echo ""
    echo "3. Start working on exercises in starter/ directory"
else
    echo -e "${YELLOW}⚠ Installation Incomplete${NC}"
    echo ""
    echo "Wait a few more minutes for pods to become ready, then run:"
    echo "  ./scripts/verify-installation.sh"
    echo ""
    echo "Or watch pod status:"
    echo "  kubectl get pods -n kubeflow -w"
fi
echo "=================================================="
