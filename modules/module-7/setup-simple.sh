#!/bin/bash
# =============================================================================
# Simple Setup Script for Module 7: CI/CD
# =============================================================================
#
# This script helps you set up the CI/CD demo by:
# 1. Checking prerequisites
# 2. Verifying kind cluster is running
# 3. Providing next steps
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Module 7: Simple CI/CD Setup${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# ==============================================================================
# Check Prerequisites
# ==============================================================================

echo -e "${YELLOW}1. Checking prerequisites...${NC}"

# Check GitHub CLI
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) is not installed${NC}"
    echo -e "   Install from: https://cli.github.com/"
    exit 1
fi
echo -e "${GREEN}✓${NC} GitHub CLI installed"

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl is not installed${NC}"
    echo -e "   Install from: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi
echo -e "${GREEN}✓${NC} kubectl installed"

# Check kind
if ! command -v kind &> /dev/null; then
    echo -e "${RED}❌ kind is not installed${NC}"
    echo -e "   Install from: https://kind.sigs.k8s.io/docs/user/quick-start/"
    exit 1
fi
echo -e "${GREEN}✓${NC} kind installed"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo -e "   Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker installed"

echo ""

# ==============================================================================
# Check kind Cluster
# ==============================================================================

echo -e "${YELLOW}2. Checking kind cluster...${NC}"

if kind get clusters 2>/dev/null | grep -q "mlops-workshop"; then
    echo -e "${GREEN}✓${NC} kind cluster 'mlops-workshop' is running"

    # Verify connectivity
    if kubectl cluster-info &> /dev/null; then
        echo -e "${GREEN}✓${NC} kubectl can connect to cluster"
    else
        echo -e "${RED}❌ kubectl cannot connect to cluster${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠${NC}  kind cluster 'mlops-workshop' not found"
    echo -e "   Creating cluster..."
    kind create cluster --name mlops-workshop
    echo -e "${GREEN}✓${NC} Cluster created"
fi

echo ""

# ==============================================================================
# Verify Git Repository
# ==============================================================================

echo -e "${YELLOW}3. Checking Git repository...${NC}"

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Not in a Git repository${NC}"
    echo -e "   Run: git init && git add . && git commit -m 'Initial commit'"
    exit 1
fi
echo -e "${GREEN}✓${NC} Git repository initialized"

# Check if GitHub remote exists
if git remote get-url origin &> /dev/null; then
    REPO_URL=$(git remote get-url origin)
    echo -e "${GREEN}✓${NC} GitHub remote configured: ${REPO_URL}"
else
    echo -e "${YELLOW}⚠${NC}  No GitHub remote configured"
    echo -e "   To push code and trigger CI/CD, you need a GitHub repository"
    echo -e "   Create one and add remote:"
    echo -e "   ${BLUE}gh repo create ml-con-workshop --public --source=. --remote=origin${NC}"
    echo ""
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo -e "1. ${YELLOW}Push code to GitHub to trigger the CI/CD workflow:${NC}"
echo -e "   ${BLUE}git add .${NC}"
echo -e "   ${BLUE}git commit -m 'Add CI/CD workflow'${NC}"
echo -e "   ${BLUE}git push origin main${NC}"
echo ""
echo -e "2. ${YELLOW}Watch the workflow run:${NC}"
echo -e "   ${BLUE}gh run watch${NC}"
echo -e "   Or visit: https://github.com/<your-username>/<your-repo>/actions"
echo ""
echo -e "3. ${YELLOW}After the workflow completes, deploy manually:${NC}"
echo -e "   Follow the deployment instructions in the GitHub Actions summary"
echo -e "   (The workflow will output exact commands to run)"
echo ""
echo -e "4. ${YELLOW}Test the deployed services:${NC}"
echo -e "   ${BLUE}curl http://localhost:30080/health${NC}"
echo -e "   ${BLUE}curl -X POST http://localhost:30080/predict \\${NC}"
echo -e "   ${BLUE}  -H 'Content-Type: application/json' \\${NC}"
echo -e "   ${BLUE}  -d '{\"request\": {\"text\": \"This workshop is great!\"}}'${NC}"
echo ""

echo -e "${YELLOW}📖 For more details, see: modules/module-7/README.md${NC}"
echo ""
