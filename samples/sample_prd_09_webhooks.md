# PRD: Webhook Delivery System

## Overview

Build a webhook system allowing customers to receive real-time HTTP callbacks when events occur in their account.

## Background

Integration requirements:
- Customers want to sync data with external systems
- Current polling approach is inefficient
- Need reliable, ordered event delivery

## Requirements

### 1. Webhook Configuration UI

Create management interface at /settings/webhooks:
- List existing webhooks (endpoint URL, events, status)
- "Add webhook" button
- Edit/delete existing webhooks
- Enable/disable toggle per webhook
- Test button to send sample payload

### 2. Webhook Setup Form

When adding/editing a webhook:
- Endpoint URL (required, must be HTTPS)
- Description (optional, for user reference)
- Event selection (checkboxes grouped by category)
- Secret key (auto-generated, user can regenerate)
- Active/inactive toggle

Validate URL is reachable before saving.

### 3. Supported Events

Deliver webhooks for these events:

**Project Events**
- project.created
- project.updated
- project.deleted
- project.archived

**Task Events**
- task.created
- task.updated
- task.completed
- task.deleted
- task.assigned
- task.comment.added

**User Events**
- user.invited
- user.joined
- user.removed
- user.role.changed

**Billing Events**
- subscription.created
- subscription.updated
- subscription.cancelled
- payment.succeeded
- payment.failed

### 4. Payload Format

Standard payload structure:
```json
{
  "id": "evt_abc123",
  "type": "task.completed",
  "created": "2024-01-15T10:30:00Z",
  "data": {
    "object": { ... }
  },
  "account_id": "acc_xyz789"
}
```

Include `X-Webhook-Signature` header with HMAC-SHA256 signature.

### 5. Delivery Mechanism

Reliable delivery system:
- Queue events in message queue (SQS/RabbitMQ)
- Deliver within 30 seconds of event
- Timeout: 30 seconds per attempt
- Retry failed deliveries: 5 attempts with exponential backoff
- Retry schedule: 1min, 5min, 30min, 2hr, 24hr
- Mark webhook as failing after 5 consecutive failures
- Alert user via email when webhook disabled

### 6. Delivery Logs

Show delivery history per webhook:
- Timestamp
- Event type
- HTTP status code
- Response time (ms)
- Request/response bodies (truncated to 10KB)
- Retry count
- Success/failure status

Retain logs for 30 days.

### 7. Manual Retry

Allow manual redelivery:
- "Retry" button on failed deliveries
- "Replay" to resend successful events
- Bulk retry option for date range

### 8. Rate Limiting

Protect destination servers:
- Max 100 events/second per endpoint
- Queue excess events
- Expose rate limit headers
- Allow customer to configure lower limits

## Success Criteria

- 99.95% delivery success rate
- p95 delivery latency < 5 seconds
- Handle 10,000 events/second platform-wide
- Zero duplicate deliveries

## Technical Notes

- Use idempotency keys to prevent duplicates
- Consider dead letter queue for persistent failures
- Implement circuit breaker per endpoint
- Log all deliveries for debugging
