# Structured Logging

## Overview

SemanticChunker emits structured JSON logs for every key operation.
These logs are easy to parse, aggregate, and forward to any monitoring
system (ELK, Datadog, Splunk, Loki, etc.).

## Enabling

```python
from chunker import SemanticChunker, ChunkerConfig

config = ChunkerConfig(
    enable_structured_logging=True,
    log_level="INFO",               # DEBUG | INFO | WARNING | ERROR
)
chunker = SemanticChunker(config)
```

## Log Levels

| Level | What is logged |
|-------|---------------|
| `DEBUG` | Everything including cache hits/misses |
| `INFO` | Chunking start/complete, language detection, timing |
| `WARNING` | Fallback activations, circuit breaker transitions |
| `ERROR` | Unrecoverable errors |

## Log Format

Every event is a single-line JSON object:

```json
{
  "timestamp": 1773355746.653,
  "event": "<event_name>",
  "component": "<ComponentName>",
  ...event-specific fields...
}
```

## Events Reference

### `chunking_start`
```json
{
  "timestamp": 1773355746.653,
  "event": "chunking_start",
  "component": "SemanticChunker",
  "doc_id": "doc123",
  "text_length": 4200,
  "strategy": "structure_aware"
}
```

### `chunking_complete`
```json
{
  "timestamp": 1773355746.900,
  "event": "chunking_complete",
  "component": "SemanticChunker",
  "doc_id": "doc123",
  "chunk_count": 8,
  "duration_ms": 247.3,
  "strategy": "structure_aware"
}
```

### `language_detected`
```json
{
  "timestamp": 1773355746.660,
  "event": "language_detected",
  "component": "LanguageDetector",
  "doc_id": "doc123",
  "language": "ar",
  "confidence": 0.94
}
```

### `operation_timing`
```json
{
  "timestamp": 1773355746.870,
  "event": "operation_timing",
  "operation": "strategy_chunking",
  "duration_ms": 123.4,
  "strategy": "semantic",
  "sentence_count": 42
}
```

### `fallback_activated`  *(WARNING)*
```json
{
  "timestamp": 1773355746.750,
  "event": "fallback_activated",
  "component": "SemanticChunker",
  "reason": "embedding provider failure",
  "fallback_action": "using recursive strategy"
}
```

### `cache_hit`  *(DEBUG)*
```json
{
  "timestamp": 1773355746.662,
  "event": "cache_hit",
  "cache_type": "embedding",
  "key_hash": "a3f1..."
}
```

### `error_occurred`  *(ERROR)*
```json
{
  "timestamp": 1773355746.900,
  "event": "error_occurred",
  "component": "SemanticChunker",
  "error_type": "EmbeddingError",
  "message": "Connection timeout",
  "recoverable": true
}
```

## Use Cases

### Performance Monitoring

```python
import json

with open("chunker.log") as f:
    for line in f:
        try:
            event = json.loads(line)
            if event["event"] == "chunking_complete":
                print(
                    f"Doc {event['doc_id']:20s} "
                    f"{event['chunk_count']:3d} chunks  "
                    f"{event['duration_ms']:6.1f} ms"
                )
        except (json.JSONDecodeError, KeyError):
            pass
```

### Detect Fallback Rate

```python
config = ChunkerConfig(
    enable_structured_logging=True,
    log_level="WARNING",    # only fallbacks & errors
)
```

Then alert if `fallback_activated` events exceed a threshold.

### Forward to Datadog / Prometheus

```python
import json
import logging

class MetricsForwarder(logging.Handler):
    def emit(self, record):
        try:
            event = json.loads(record.getMessage())
            if event["event"] == "chunking_complete":
                datadog.histogram("chunker.duration_ms", event["duration_ms"])
                datadog.increment("chunker.chunks", event["chunk_count"])
            elif event["event"] == "fallback_activated":
                datadog.increment("chunker.fallbacks")
        except (json.JSONDecodeError, KeyError):
            pass

logging.getLogger("chunker").addHandler(MetricsForwarder())
```

### Cache Performance Analysis

```python
config = ChunkerConfig(
    enable_structured_logging=True,
    log_level="DEBUG",
    enable_embedding_cache=True,
)
# Parse cache_hit events to track hit rates over time
```

## Best Practices

1. **Use `INFO` in production** — good signal-to-noise ratio
2. **Use `DEBUG` when troubleshooting** — includes cache and timing details
3. **Parse logs asynchronously** — don't block the chunking pipeline
4. **Monitor fallback rates** — a high rate may mean a misconfigured embedding provider
5. **Track `duration_ms` trends** — to detect performance degradation over time

## See Also

- [examples/structured_logging_example.py](../examples/structured_logging_example.py)
- [examples/metrics_example.py](../examples/metrics_example.py)
- [User Guide — Monitoring section](user_guide.md#monitoring--observability)
