#!/bin/bash

# =============================================================================
# Module 3 - Kubernetes Deployment Script
# =============================================================================
#
# This script helps you deploy the sentiment API to Kubernetes
# Supports all 4 steps with explanations
#
# Usage:
#   ./deploy.sh               # Interactive mode
#   ./deploy.sh step1         # Deploy specific step
#   ./deploy.sh step2
#   ./deploy.sh step3
#   ./deploy.sh step4
#
# Prerequisites:
#   - kind cluster running (kind get clusters)
#   - kubectl configured (kubectl cluster-info)
#   - Docker image built (sentiment-api:v1)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl first."
        exit 1
    fi
    print_success "kubectl found"

    # Check kind
    if ! command -v kind &> /dev/null; then
        print_error "kind not found. Please install kind first."
        exit 1
    fi
    print_success "kind found"

    # Check if cluster exists
    if ! kind get clusters | grep -q "mlops-workshop"; then
        print_error "kind cluster 'mlops-workshop' not found."
        print_info "Create it with: kind create cluster --name mlops-workshop"
        exit 1
    fi
    print_success "kind cluster 'mlops-workshop' found"

    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster."
        print_info "Switch context with: kubectl config use-context kind-mlops-workshop"
        exit 1
    fi
    print_success "kubectl connected to cluster"

    # Check if Docker image exists
    if ! docker images | grep -q "sentiment-api.*v1"; then
        print_warning "Docker image 'sentiment-api:v1' not found locally."
        print_info "Build it with: cd ../module-2 && bentoml containerize sentiment_service:latest -t sentiment-api:v1"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Docker image 'sentiment-api:v1' found"
    fi

    # Check if image is loaded in kind
    print_info "Checking if image is loaded in kind cluster..."
    if ! docker exec mlops-workshop-control-plane crictl images | grep -q "sentiment-api"; then
        print_warning "Docker image not loaded in kind cluster."
        print_info "Loading image into kind cluster..."
        if kind load docker-image sentiment-api:v1 --name mlops-workshop; then
            print_success "Image loaded into kind cluster"
        else
            print_error "Failed to load image into kind cluster"
            exit 1
        fi
    else
        print_success "Docker image already loaded in kind cluster"
    fi

    # For step4, check if metrics-server is installed
    if [[ "$1" == "step4" ]]; then
        print_info "Checking metrics-server (required for HPA)..."
        if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
            print_warning "metrics-server not found (required for HPA)"
            print_info "Installing metrics-server..."

            kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

            # Patch for kind (disable TLS verification)
            kubectl patch deployment metrics-server -n kube-system --type='json' \
                -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'

            print_info "Waiting for metrics-server to be ready..."
            kubectl wait --for=condition=available --timeout=60s deployment/metrics-server -n kube-system

            print_success "metrics-server installed and ready"
        else
            print_success "metrics-server already installed"
        fi
    fi
}

# Deploy a specific step
deploy_step() {
    local step=$1
    local yaml_file="step${step#step}-*.yaml"

    # Find the actual file
    local file=$(ls step${step#step}-*.yaml 2>/dev/null | head -1)

    if [[ -z "$file" ]]; then
        print_error "File not found: step${step#step}-*.yaml"
        exit 1
    fi

    print_header "Deploying $step: $file"

    # Show what will be deployed
    print_info "Resources to be created:"
    kubectl apply -f "$file" --dry-run=client | grep -E "^(configmap|deployment|service|horizontalpodautoscaler|poddisruptionbudget)" || true

    # Ask for confirmation
    read -p "Deploy these resources? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled"
        exit 0
    fi

    # Deploy
    print_info "Deploying..."
    if kubectl apply -f "$file"; then
        print_success "Resources applied successfully"
    else
        print_error "Failed to apply resources"
        exit 1
    fi

    # Wait for deployment to be ready
    print_info "Waiting for deployment to be ready..."
    if kubectl wait --for=condition=available --timeout=300s deployment/sentiment-api; then
        print_success "Deployment is ready"
    else
        print_error "Deployment failed to become ready"
        print_info "Check status with: kubectl get pods -l app=sentiment-api"
        print_info "Check logs with: kubectl logs -l app=sentiment-api"
        exit 1
    fi

    # Show deployment status
    print_header "Deployment Status"
    kubectl get all -l app=sentiment-api

    # Show step-specific information
    case "$step" in
        step1)
            print_header "Step 1: Basic Deployment"
            print_info "You've deployed the simplest possible Kubernetes deployment!"
            print_info ""
            print_info "What was created:"
            print_info "  • Deployment with 2 replicas"
            print_info "  • NodePort Service on port 30080"
            print_info ""
            print_info "What's missing (will add in next steps):"
            print_info "  ❌ No resource limits"
            print_info "  ❌ No health checks"
            print_info "  ❌ No ConfigMap"
            print_info "  ❌ No auto-scaling"
            ;;
        step2)
            print_header "Step 2: Resource Management"
            print_info "You've added resource limits and requests!"
            print_info ""
            print_info "New features:"
            print_info "  ✅ CPU requests: 500m, limits: 1000m"
            print_info "  ✅ Memory requests: 1Gi, limits: 2Gi"
            print_info "  ✅ QoS Class: Burstable"
            print_info ""
            print_info "Check resource usage:"
            print_info "  kubectl top pod -l app=sentiment-api"
            ;;
        step3)
            print_header "Step 3: Health Checks & ConfigMap"
            print_info "You've added health probes and externalized configuration!"
            print_info ""
            print_info "New features:"
            print_info "  ✅ Startup probe (handles slow model loading)"
            print_info "  ✅ Liveness probe (restarts dead containers)"
            print_info "  ✅ Readiness probe (controls traffic)"
            print_info "  ✅ ConfigMap for environment variables"
            print_info ""
            print_info "Check probe status:"
            print_info "  kubectl describe pod <pod-name> | grep -A 5 Probes"
            ;;
        step4)
            print_header "Step 4: Production-Ready Deployment"
            print_info "You've deployed a COMPLETE production setup!"
            print_info ""
            print_info "New features:"
            print_info "  ✅ Horizontal Pod Autoscaler (2-10 replicas)"
            print_info "  ✅ Pod Disruption Budget (min 1 available)"
            print_info "  ✅ Security context (non-root, read-only)"
            print_info "  ✅ Pod anti-affinity (spread across nodes)"
            print_info ""
            print_info "Check HPA status:"
            print_info "  kubectl get hpa sentiment-api-hpa"
            print_info ""
            print_info "Check PDB status:"
            print_info "  kubectl get pdb sentiment-api-pdb"
            ;;
    esac

    # Test instructions
    print_header "Testing the Deployment"
    print_info "In one terminal, forward the port:"
    print_info "  kubectl port-forward svc/sentiment-api-service 8080:80"
    print_info ""
    print_info "In another terminal, test the API:"
    print_info "  curl -X POST http://localhost:8080/predict \\"
    print_info "       -H 'Content-Type: application/json' \\"
    print_info "       -d '{\"text\": \"Kubernetes is amazing!\"}'"
    print_info ""
    print_info "Check logs:"
    print_info "  kubectl logs -l app=sentiment-api --tail=50 -f"
}

# Interactive menu
show_menu() {
    print_header "Module 3 - Kubernetes Deployment"
    echo "Select which step to deploy:"
    echo ""
    echo "  1) Step 1 - Basic Deployment"
    echo "     └─ Simple deployment with 2 replicas + Service"
    echo ""
    echo "  2) Step 2 - Add Resource Limits"
    echo "     └─ Everything from step 1 + resource management"
    echo ""
    echo "  3) Step 3 - Add Health Probes & ConfigMap"
    echo "     └─ Everything from step 2 + probes + ConfigMap"
    echo ""
    echo "  4) Step 4 - Production-Ready (Recommended)"
    echo "     └─ Everything from step 3 + HPA + PDB + Security"
    echo ""
    echo "  5) Exit"
    echo ""
    read -p "Enter choice [1-5]: " choice

    case $choice in
        1)
            check_prerequisites "step1"
            deploy_step "step1"
            ;;
        2)
            check_prerequisites "step2"
            deploy_step "step2"
            ;;
        3)
            check_prerequisites "step3"
            deploy_step "step3"
            ;;
        4)
            check_prerequisites "step4"
            deploy_step "step4"
            ;;
        5)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Main script
main() {
    if [[ $# -eq 0 ]]; then
        # Interactive mode
        show_menu
    else
        # Command-line mode
        local step=$1
        if [[ ! "$step" =~ ^step[1-4]$ ]]; then
            print_error "Invalid step: $step"
            print_info "Usage: $0 [step1|step2|step3|step4]"
            exit 1
        fi
        check_prerequisites "$step"
        deploy_step "$step"
    fi
}

# Run main function
main "$@"
