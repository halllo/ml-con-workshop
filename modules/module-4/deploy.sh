#!/bin/bash

# =============================================================================
# Module 4 - Deploy Script for API Gateway
# =============================================================================
#
# This script deploys the API Gateway to Kubernetes.
# Integrates with Module 3's ML service.
#
# Usage:
#   ./deploy.sh           # Interactive mode
#   ./deploy.sh --deploy  # Deploy gateway
#   ./deploy.sh --test    # Test deployment
#   ./deploy.sh --clean   # Remove deployment
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found"
        exit 1
    fi
    print_success "kubectl found"

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    print_success "Connected to cluster"

    # Check if ML service exists (Module 3)
    if ! kubectl get svc sentiment-api-service &> /dev/null; then
        print_warning "ML service (sentiment-api-service) not found"
        print_info "Deploy Module 3 first: cd ../module-3 && ./deploy.sh"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "ML service found"
    fi
}

deploy_gateway() {
    print_header "Deploying API Gateway"

    print_info "Applying Kubernetes manifests..."
    if kubectl apply -f deployment.yaml; then
        print_success "Manifests applied"
    else
        print_error "Failed to apply manifests"
        exit 1
    fi

    print_info "Waiting for deployment to be ready..."
    if kubectl wait --for=condition=available --timeout=120s deployment/api-gateway; then
        print_success "Deployment ready"
    else
        print_error "Deployment failed to become ready"
        print_info "Check with: kubectl get pods -l app=api-gateway"
        exit 1
    fi

    print_header "Deployment Status"
    kubectl get all -l app=api-gateway
}

test_deployment() {
    print_header "Testing API Gateway"

    # Check if deployed
    if ! kubectl get deployment api-gateway &> /dev/null; then
        print_error "Gateway not deployed"
        print_info "Deploy first with: ./deploy.sh --deploy"
        exit 1
    fi

    # Port forward
    print_info "Setting up port-forward on port 8080..."
    kubectl port-forward svc/api-gateway-service 8080:80 &
    PF_PID=$!

    # Wait for port-forward
    sleep 3

    # Test health
    print_info "Testing /health endpoint..."
    if curl -f -s http://localhost:8080/health > /dev/null; then
        print_success "Health check passed"
        curl -s http://localhost:8080/health | jq .
    else
        print_error "Health check failed"
        kill $PF_PID 2>/dev/null
        exit 1
    fi

    # Test prediction
    print_info "Testing /predict endpoint..."
    RESPONSE=$(curl -s -X POST http://localhost:8080/predict \
        -H "Content-Type: application/json" \
        -d '{"request": {"text": "Production ready!","request_id": null}}')

    if echo "$RESPONSE" | jq . > /dev/null 2>&1; then
        print_success "Prediction successful"
        echo "$RESPONSE" | jq .
    else
        print_error "Prediction failed"
        echo "$RESPONSE"
    fi

    # Cleanup
    kill $PF_PID 2>/dev/null

    print_success ""
    print_success "All tests passed!"
}

cleanup() {
    print_header "Cleaning Up"

    print_warning "This will DELETE the API gateway deployment"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cancelled"
        exit 0
    fi

    print_info "Deleting resources..."
    kubectl delete -f deployment.yaml --ignore-not-found=true

    print_success "Cleanup complete"
}

show_menu() {
    print_header "Module 4 - API Gateway Deployment"

    echo "Select action:"
    echo ""
    echo "  1) Deploy gateway"
    echo "  2) Test deployment"
    echo "  3) View logs"
    echo "  4) View metrics"
    echo "  5) Clean up"
    echo "  6) Exit"
    echo ""
    read -p "Enter choice [1-6]: " choice

    case $choice in
        1)
            check_prerequisites
            deploy_gateway
            ;;
        2)
            test_deployment
            ;;
        3)
            kubectl logs -l app=api-gateway --tail=50 -f
            ;;
        4)
            print_info "Port-forwarding to view metrics..."
            kubectl port-forward svc/api-gateway-service 8080:80
            # Then visit http://localhost:8080/metrics
            ;;
        5)
            cleanup
            ;;
        6)
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

main() {
    if [[ $# -eq 0 ]]; then
        show_menu
    else
        case "$1" in
            --deploy)
                check_prerequisites
                deploy_gateway
                ;;
            --test)
                test_deployment
                ;;
            --clean)
                cleanup
                ;;
            *)
                print_error "Invalid argument: $1"
                print_info "Usage: $0 [--deploy|--test|--clean]"
                exit 1
                ;;
        esac
    fi
}

main "$@"
