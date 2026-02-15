# Multi-stage build for PRD Decomposer MCP Server
# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (without dev dependencies)
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/
COPY samples/ ./samples/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Default to stdio transport (can be overridden to "http")
ENV MCP_TRANSPORT="stdio"

# Health check disabled by default (stdio mode doesn't support HTTP probes)
# For HTTP mode deployments, override with:
#   HEALTHCHECK CMD python -c "from prd_decomposer.server import health_check; result = health_check(); exit(0 if result['status'] == 'healthy' else 1)"
HEALTHCHECK NONE

# Run the MCP server
ENTRYPOINT ["python", "-m", "prd_decomposer.server"]
