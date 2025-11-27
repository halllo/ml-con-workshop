// Module 4: Production-Ready API Gateway
// ========================================
//
// In this exercise, you'll build a complete production-ready API Gateway in Go.
// You'll implement ALL production features in one comprehensive exercise.
//
// Time: ~50 minutes | TODOs: 20
//
// What you'll learn:
// - HTTP server configuration in Go
// - Reverse proxy pattern
// - Structured logging with log/slog
// - Prometheus metrics
// - Request ID tracking
// - Middleware pattern
// - Graceful shutdown
// - Backend health checks
//
// What you'll implement:
// - Complete HTTP server setup (TODOs 1-6)
// - Health and proxy handlers (TODOs 7-10)
// - Structured logging (TODOs 11-12)
// - Prometheus metrics (TODOs 13-14)
// - Middleware pattern (TODOs 15-17)
// - Request proxying (TODOs 18-19)
// - Graceful shutdown (TODO 20)

package main

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// =============================================================================
// CONFIGURATION
// =============================================================================

type Config struct {
	Port           string
	BackendURL     string
	RequestTimeout time.Duration
	LogLevel       string
	Environment    string
}

func loadConfig() *Config {
	getEnv := func(key, defaultValue string) string {
		if value := os.Getenv(key); value != "" {
			return value
		}
		return defaultValue
	}

	// TODO 1: Complete configuration loading
	// FILL IN: Return Config struct with all fields
	// HINT:
	//   Port: getEnv("GATEWAY_PORT", "8080")
	//   BackendURL: getEnv("BACKEND_URL", "http://sentiment-api-service:80")
	//   RequestTimeout: 10 * time.Second
	//   LogLevel: getEnv("LOG_LEVEL", "info")
	//   Environment: getEnv("ENVIRONMENT", "production")
	// YOUR CODE HERE
	return nil // REPLACE THIS
}

// =============================================================================
// PROMETHEUS METRICS
// =============================================================================

var (
	// TODO 2: Define HTTP request counter
	// FILL IN: Use promauto.NewCounterVec with prometheus.CounterOpts
	// HINT:
	//   Name: "gateway_http_requests_total"
	//   Help: "Total number of HTTP requests"
	//   Labels: []string{"method", "endpoint", "status"}
	// YOUR CODE HERE
	httpRequestsTotal = nil // REPLACE WITH: promauto.NewCounterVec(...)

	// TODO 3: Define HTTP request duration histogram
	// FILL IN: Use promauto.NewHistogramVec with prometheus.HistogramOpts
	// HINT:
	//   Name: "gateway_http_request_duration_seconds"
	//   Help: "HTTP request duration in seconds"
	//   Buckets: prometheus.DefBuckets
	//   Labels: []string{"method", "endpoint"}
	// YOUR CODE HERE
	httpRequestDuration = nil // REPLACE WITH: promauto.NewHistogramVec(...)

	backendRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gateway_backend_requests_total",
			Help: "Total number of backend requests",
		},
		[]string{"endpoint", "status"},
	)

	backendRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "gateway_backend_request_duration_seconds",
			Help:    "Backend request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"endpoint"},
	)
)

// =============================================================================
// MIDDLEWARE
// =============================================================================

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
	written    int64
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	n, err := rw.ResponseWriter.Write(b)
	rw.written += int64(n)
	return n, err
}

// Request ID middleware - adds unique ID to each request
func requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// TODO 4: Generate and add request ID
		// FILL IN: Check for existing X-Request-ID, generate UUID if not present
		// HINT:
		//   requestID := r.Header.Get("X-Request-ID")
		//   if requestID == "" {
		//       requestID = uuid.New().String()
		//   }
		//   ctx := context.WithValue(r.Context(), "request_id", requestID)
		//   w.Header().Set("X-Request-ID", requestID)
		//   next.ServeHTTP(w, r.WithContext(ctx))
		// YOUR CODE HERE

	})
}

// Logging middleware - logs all requests
func loggingMiddleware(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()

			rw := &responseWriter{
				ResponseWriter: w,
				statusCode:     http.StatusOK,
			}

			requestID, _ := r.Context().Value("request_id").(string)

			logger.Info("incoming request",
				"request_id", requestID,
				"method", r.Method,
				"path", r.URL.Path,
				"remote_addr", r.RemoteAddr,
			)

			// TODO 5: Call next handler and log completion
			// FILL IN: Call next.ServeHTTP, then log completion with duration
			// HINT:
			//   next.ServeHTTP(rw, r)
			//   duration := time.Since(start)
			//   logger.Info("request completed",
			//       "request_id", requestID,
			//       "method", r.Method,
			//       "path", r.URL.Path,
			//       "status", rw.statusCode,
			//       "duration_ms", duration.Milliseconds(),
			//       "bytes_written", rw.written)
			// YOUR CODE HERE

		})
	}
}

// Metrics middleware - records Prometheus metrics
func metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		rw := &responseWriter{
			ResponseWriter: w,
			statusCode:     http.StatusOK,
		}

		next.ServeHTTP(rw, r)

		duration := time.Since(start).Seconds()

		// TODO 6: Record metrics
		// FILL IN: Increment counter and observe duration
		// HINT:
		//   httpRequestsTotal.WithLabelValues(r.Method, r.URL.Path, fmt.Sprintf("%d", rw.statusCode)).Inc()
		//   httpRequestDuration.WithLabelValues(r.Method, r.URL.Path).Observe(duration)
		// YOUR CODE HERE

	})
}

// Chain multiple middlewares
func chain(handler http.Handler, middlewares ...func(http.Handler) http.Handler) http.Handler {
	for i := len(middlewares) - 1; i >= 0; i-- {
		handler = middlewares[i](handler)
	}
	return handler
}

// =============================================================================
// GATEWAY
// =============================================================================

type Gateway struct {
	config     *Config
	logger     *slog.Logger
	httpClient *http.Client
}

func NewGateway(config *Config, logger *slog.Logger) *Gateway {
	return &Gateway{
		config: config,
		logger: logger,
		httpClient: &http.Client{
			Timeout: config.RequestTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

func (g *Gateway) createHandler() http.Handler {
	// TODO 7: Create HTTP mux and register routes
	// FILL IN: Create mux, register all routes, apply middleware
	// HINT:
	//   mux := http.NewServeMux()
	//   mux.HandleFunc("/health", g.handleHealth)
	//   mux.HandleFunc("/predict", g.handlePredict)
	//   mux.HandleFunc("/batch_predict", g.handleBatchPredict)
	//   mux.Handle("/metrics", promhttp.Handler())
	//   mux.HandleFunc("/", g.handleRoot)
	//   return chain(mux, requestIDMiddleware, loggingMiddleware(g.logger), metricsMiddleware)
	// YOUR CODE HERE
	return nil // REPLACE THIS
}

func (g *Gateway) handleHealth(w http.ResponseWriter, r *http.Request) {
	// TODO 8: Validate HTTP method
	// FILL IN: Check if method is GET, return 405 if not
	// HINT:
	//   if r.Method != http.MethodGet {
	//       http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	//       return
	//   }
	// YOUR CODE HERE

	// TODO 9: Test backend connectivity
	// FILL IN: Make request to backend /health endpoint with timeout
	// HINT:
	//   ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	//   defer cancel()
	//   req, _ := http.NewRequestWithContext(ctx, http.MethodGet, g.config.BackendURL+"/health", nil)
	//   resp, err := g.httpClient.Do(req)
	// YOUR CODE HERE

	health := map[string]interface{}{
		"status":  "ok",
		"service": "api-gateway",
	}

	// TODO 10: Add backend health status to response
	// FILL IN: Check if backend request succeeded, add to health map
	// HINT:
	//   if err != nil {
	//       health["backend"] = "unreachable"
	//       health["backend_error"] = err.Error()
	//   } else {
	//       resp.Body.Close()
	//       health["backend"] = "ok"
	//       health["backend_status"] = resp.StatusCode
	//   }
	// YOUR CODE HERE

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(health)
}

func (g *Gateway) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.Error(w, "Not found", http.StatusNotFound)
		return
	}

	info := map[string]interface{}{
		"service":     "Sentiment Analysis API Gateway",
		"version":     "2.0.0",
		"environment": g.config.Environment,
		"endpoints": map[string]string{
			"/health":        "Health check endpoint",
			"/predict":       "Single text sentiment analysis",
			"/batch_predict": "Batch sentiment analysis",
			"/metrics":       "Prometheus metrics",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}

func (g *Gateway) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	g.proxyRequest(w, r, "/predict")
}

func (g *Gateway) handleBatchPredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	g.proxyRequest(w, r, "/batch_predict")
}

func (g *Gateway) proxyRequest(w http.ResponseWriter, r *http.Request, path string) {
	requestID, _ := r.Context().Value("request_id").(string)

	// TODO 11: Read request body
	// FILL IN: Read and validate request body
	// HINT:
	//   body, err := io.ReadAll(r.Body)
	//   if err != nil {
	//       g.logger.Error("failed to read request body", "request_id", requestID, "error", err)
	//       http.Error(w, "Failed to read request", http.StatusBadRequest)
	//       return
	//   }
	//   defer r.Body.Close()
	// YOUR CODE HERE

	// TODO 12: Create backend request
	// FILL IN: Create HTTP request to backend service
	// HINT:
	//   backendURL := g.config.BackendURL + path
	//   req, err := http.NewRequestWithContext(r.Context(), r.Method, backendURL, bytes.NewReader(body))
	//   if err != nil { ... }
	// YOUR CODE HERE

	// TODO 13: Copy headers and add request ID
	// FILL IN: Copy headers from incoming request to backend request
	// HINT:
	//   for key, values := range r.Header {
	//       if key != "Host" && key != "Connection" {
	//           for _, value := range values {
	//               req.Header.Add(key, value)
	//           }
	//       }
	//   }
	//   req.Header.Set("X-Request-ID", requestID)
	// YOUR CODE HERE

	// TODO 14: Send request to backend and record metrics
	// FILL IN: Execute request, record duration, handle errors
	// HINT:
	//   start := time.Now()
	//   resp, err := g.httpClient.Do(req)
	//   duration := time.Since(start).Seconds()
	//   statusLabel := "error"
	//   if resp != nil {
	//       statusLabel = fmt.Sprintf("%d", resp.StatusCode)
	//   }
	//   backendRequestsTotal.WithLabelValues(path, statusLabel).Inc()
	//   backendRequestDuration.WithLabelValues(path).Observe(duration)
	//   if err != nil { ... }
	//   defer resp.Body.Close()
	// YOUR CODE HERE

	// TODO 15: Copy response headers to client
	// FILL IN: Copy all response headers
	// HINT:
	//   for key, values := range resp.Header {
	//       for _, value := range values {
	//           w.Header().Add(key, value)
	//       }
	//   }
	// YOUR CODE HERE

	// TODO 16: Copy status code and body to client
	// FILL IN: Forward response to client
	// HINT:
	//   w.WriteHeader(resp.StatusCode)
	//   _, err = io.Copy(w, resp.Body)
	//   if err != nil {
	//       g.logger.Error("failed to write response", "request_id", requestID, "error", err)
	//   }
	// YOUR CODE HERE
}

// =============================================================================
// MAIN
// =============================================================================

func main() {
	config := loadConfig()

	// TODO 17: Setup structured logging
	// FILL IN: Create JSON logger with appropriate log level
	// HINT:
	//   var logLevel slog.Level
	//   switch config.LogLevel {
	//   case "debug": logLevel = slog.LevelDebug
	//   case "info": logLevel = slog.LevelInfo
	//   case "warn": logLevel = slog.LevelWarn
	//   case "error": logLevel = slog.LevelError
	//   default: logLevel = slog.LevelInfo
	//   }
	//   logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: logLevel}))
	//   slog.SetDefault(logger)
	// YOUR CODE HERE
	logger := nil // REPLACE THIS

	logger.Info("starting api gateway",
		"port", config.Port,
		"backend_url", config.BackendURL,
		"environment", config.Environment,
	)

	gateway := NewGateway(config, logger)

	// TODO 18: Create HTTP server
	// FILL IN: Configure HTTP server with timeouts
	// HINT:
	//   server := &http.Server{
	//       Addr:         ":" + config.Port,
	//       Handler:      gateway.createHandler(),
	//       ReadTimeout:  15 * time.Second,
	//       WriteTimeout: 15 * time.Second,
	//       IdleTimeout:  60 * time.Second,
	//   }
	// YOUR CODE HERE
	server := nil // REPLACE THIS

	// TODO 19: Start server in goroutine
	// FILL IN: Start server asynchronously
	// HINT:
	//   go func() {
	//       logger.Info("server listening", "addr", server.Addr)
	//       if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
	//           logger.Error("server failed", "error", err)
	//           os.Exit(1)
	//       }
	//   }()
	// YOUR CODE HERE

	// TODO 20: Graceful shutdown
	// FILL IN: Wait for SIGINT/SIGTERM, then shutdown gracefully
	// HINT:
	//   quit := make(chan os.Signal, 1)
	//   signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	//   <-quit
	//   logger.Info("shutting down server...")
	//   ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	//   defer cancel()
	//   if err := server.Shutdown(ctx); err != nil {
	//       logger.Error("server forced to shutdown", "error", err)
	//       os.Exit(1)
	//   }
	//   logger.Info("server stopped gracefully")
	// YOUR CODE HERE
}
