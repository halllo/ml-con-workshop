# Load podman image with kind

```bash
kind load docker-image localhost/sentiment-api:v1 --name mlops-workshop
```

This fails:

```log
enabling experimental podman provider
ERROR: image: "localhost/sentiment-api:v1" not present locally
```

<https://github.com/kubernetes-sigs/kind/issues/2038>

We use `deepaiorg/sentiment-analysis` instead to continue the course.

However this image does not seem to log anything and also does not respond on port 3000 and `/healthz` endpoint:

| Type | Reason | Age | From | Message |
|------|--------|-----|------|---------|
| Normal | Scheduled | 10m | default-scheduler | Successfully assigned default/sentiment-api-64d655b7b5-6686p to mlops-workshop-worker |
| Normal | Pulling | 10m | kubelet | Pulling image "deepaiorg/sentiment-analysis" |
| Normal | Pulled | 5m3s | kubelet | Successfully pulled image "deepaiorg/sentiment-analysis" in 5m52.063s (5m52.063s including waiting). Image size: 1302830843 bytes. |
| Normal | Created | 5m2s | kubelet | Created container: sentiment-api |
| Normal | Started | 5m2s | kubelet | Started container sentiment-api |
| Warning | Unhealthy | 2m50s (x25 over 4m50s) | kubelet | Startup probe failed: Get "http://10.244.1.2:3000/healthz": dial tcp 10.244.1.2:3000: connect: connection refused |

Loading the images with podman directly tells us that it listens on port 5000. It does not respond at `/healthz` nor `/health`.

Course instructor pushed a new image: <https://hub.docker.com/r/rfashwal/sentiment_service>.
