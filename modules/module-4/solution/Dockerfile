# Multi-Stage Docker Build for Go API Gateway
# ============================================
#
# This Dockerfile demonstrates Go best practices:
# - Multi-stage build (smaller final image)
# - Separate build and runtime stages
# - Non-root user (security)
# - Minimal base image (distroless)
#
# Final image size: ~25MB (vs ~800MB with full Go image!)

# =============================================================================
# STAGE 1: BUILD
# =============================================================================
# Use official Golang image as build environment
# This image has all build tools but is large (~800MB)
FROM golang:1.21-alpine AS builder

# Install build dependencies
# - git: for fetching Go modules from git repositories
# - ca-certificates: for HTTPS requests during build
RUN apk add --no-cache git ca-certificates

# Set working directory
WORKDIR /build

# Copy go.mod and go.sum first (layer caching optimization)
# If these files don't change, Docker reuses the cached layer
# This makes rebuilds faster when only code changes
COPY go.mod go.sum ./

# Download dependencies
# This layer is cached unless go.mod/go.sum changes
RUN go mod download

# Copy source code
COPY gateway_solution.go ./

# Build the binary
# Flags explained:
#   -o gateway          - Output filename
#   -ldflags="-w -s"    - Strip debug info (smaller binary)
#   -trimpath           - Remove file system paths (security)
#   CGO_ENABLED=0       - Static binary (no C dependencies)
#   GOOS=linux          - Target OS
#   GOARCH=amd64        - Target architecture
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s" \
    -trimpath \
    -o gateway \
    gateway_solution.go

# =============================================================================
# STAGE 2: RUNTIME
# =============================================================================
# Use minimal base image
# gcr.io/distroless/static-debian11 contains:
# - Only essential runtime files
# - CA certificates (for HTTPS)
# - No shell, no package manager (more secure)
# - Size: ~2MB
FROM gcr.io/distroless/static-debian11

# Add labels for metadata
LABEL maintainer="mlops-workshop" \
      description="API Gateway for Sentiment Analysis Service" \
      version="2.0.0"

# Copy CA certificates from builder (for HTTPS requests to backend)
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy the binary from builder stage
COPY --from=builder /build/gateway /gateway

# Use non-root user (security best practice)
# distroless images provide nonroot user (UID 65532)
USER nonroot:nonroot

# Expose port
# This is documentation only - doesn't actually publish the port
EXPOSE 8080


# Run the binary
# distroless images use ENTRYPOINT instead of CMD
ENTRYPOINT ["/gateway"]
