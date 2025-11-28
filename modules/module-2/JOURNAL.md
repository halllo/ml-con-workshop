# bentoml with podman

```bash
bentoml build service_with_validation.py
bentoml list
entoml containerize sentiment_service:ky2f3cgmis4epfti -t sentiment-api:v1 --backend podman
```

See <https://github.com/bentoml/BentoML/issues/3575>
