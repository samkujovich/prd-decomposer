# PRD: API Rate Limiting System

## Overview

Implement rate limiting for our public API to prevent abuse, ensure fair usage across customers, and protect backend services from overload.

## Background

Our API currently has no rate limiting, which has led to:
- Occasional service degradation from high-volume users
- Difficulty identifying and blocking abusive clients
- No visibility into per-customer usage patterns

## Requirements

### 1. Tier-Based Rate Limits

Implement different rate limits based on customer tier:

- **Free tier**: 100 requests per minute, 1,000 requests per day
- **Pro tier**: 1,000 requests per minute, unlimited daily requests
- **Enterprise tier**: Custom limits configured per customer

The system should be fast and scalable to handle our growing traffic.

### 2. Rate Limit Response Headers

All API responses must include rate limit information:

- `X-RateLimit-Limit`: Maximum requests allowed in the window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when the window resets

### 3. Rate Limit Exceeded Handling

When a client exceeds their rate limit:

- Return HTTP 429 (Too Many Requests) status code
- Include `Retry-After` header with seconds until reset
- Return JSON error body with clear message and documentation link

### 4. Developer Dashboard

Provide a self-service dashboard where developers can:

- View their current usage and limits
- See historical usage graphs
- The dashboard should be user-friendly and intuitive

### 5. Backend Implementation

Technical requirements for the rate limiting service:

- Use Redis for storing rate limit counters (sliding window algorithm)
- Support configurable limits per endpoint (some endpoints may have stricter limits)
- Implement bypass mechanism for internal services (authenticated via service tokens)
- Logging and alerting when customers approach or exceed limits

### 6. Monitoring and Observability

- Emit metrics for rate limit checks, passes, and rejections
- Create dashboards showing rate limit health across the fleet
- Alert on anomalies (sudden spikes in rejections, Redis failures)

## Success Criteria

- Rate limiting deployed to all API endpoints
- Less than 1ms p99 latency overhead
- Zero false positives for internal services
- Dashboard accessible to all API customers

## Timeline

Phase 1: Core rate limiting (backend + headers)
Phase 2: Dashboard
Phase 3: Advanced monitoring

## Open Questions

- Should we implement graduated throttling (slow down before hard block)?
- How do we handle customers who legitimately need temporary limit increases?
