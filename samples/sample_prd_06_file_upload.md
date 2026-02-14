# PRD: File Upload Service

## Overview

Build a robust file upload service supporting large files, multiple file types, and batch uploads with progress tracking.

## Background

Current upload system limitations:
- Max file size: 10MB
- Single file upload only
- No progress indicator
- Uploads fail silently on timeout

## Requirements

### 1. Drag-and-Drop Upload Zone

Create a modern upload interface:
- Large drop zone with dashed border
- Visual feedback on drag hover (border color change, background highlight)
- Click to open file picker alternative
- Support selecting multiple files
- Show file type icons based on extension

### 2. File Validation

Validate files before upload:
- Check file size against limit (configurable per file type)
- Validate file extension against allowed list
- Verify MIME type matches extension (prevent extension spoofing)
- Scan for malware using ClamAV integration
- Show clear error message for rejected files

File type limits:
- Images (jpg, png, gif, webp): 25MB max
- Documents (pdf, doc, docx, xls, xlsx): 50MB max
- Videos (mp4, mov, avi): 500MB max
- Archives (zip, tar.gz): 100MB max

### 3. Chunked Upload

Implement chunked upload for large files:
- Split files into 5MB chunks
- Upload chunks in parallel (max 3 concurrent)
- Support resume on connection failure
- Store partial uploads for 24 hours
- Show per-chunk progress

### 4. Progress Tracking

Display upload progress:
- Overall progress bar for batch
- Individual progress per file
- Upload speed indicator (MB/s)
- Estimated time remaining
- Cancel button per file and for entire batch

### 5. Post-Upload Processing

After successful upload:
- Generate thumbnails for images (150x150, 300x300, 600x600)
- Extract metadata (dimensions, duration, page count)
- Generate preview for documents (first page as image)
- Run OCR on PDF/images if text extraction enabled
- Store in CDN with geographic distribution

### 6. Upload API

Provide programmatic upload API:
- POST /api/v1/uploads/initiate - Get upload URL and ID
- PUT /api/v1/uploads/{id}/chunks/{n} - Upload chunk
- POST /api/v1/uploads/{id}/complete - Finalize upload
- GET /api/v1/uploads/{id}/status - Check progress
- DELETE /api/v1/uploads/{id} - Cancel upload

All endpoints require authentication via API key or OAuth token.

### 7. Storage Integration

Support multiple storage backends:
- AWS S3 (primary)
- Google Cloud Storage (backup)
- Local filesystem (development)

Generate signed URLs for secure downloads with expiration.

## Success Criteria

- Support files up to 5GB
- Upload speed: saturate user's bandwidth
- 99.9% upload success rate
- Resume work for connections dropped within 24 hours
- p99 latency for initiate endpoint < 100ms

## Security Requirements

- Encrypt files at rest (AES-256)
- Encrypt in transit (TLS 1.3)
- Signed URLs expire after 1 hour
- Rate limit: 100 uploads per hour per user
- Virus scan all uploads before making available
