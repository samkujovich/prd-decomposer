# PRD: Global Search

## Overview

Implement a powerful global search feature that helps users quickly find any content across the platform.

## Background

User feedback indicates:
- Hard to find old projects and documents
- No way to search within file contents
- Navigation-heavy workflow to locate items

## Requirements

### 1. Search Bar

Add global search bar to header:
- Prominent search icon that expands to input field
- Keyboard shortcut: Cmd/Ctrl + K to focus
- Placeholder text: "Search projects, tasks, files..."
- Recent searches dropdown (last 5)
- Clear button when text entered

### 2. Instant Results

Show results as user types:
- Begin searching after 2 characters
- Debounce input by 200ms
- Show loading indicator during search
- Display results in categorized sections
- Highlight matching text in results
- Results should appear quickly

### 3. Search Scope

Search across all content types:
- **Projects**: Name, description
- **Tasks**: Title, description, comments
- **Files**: Filename, extracted text content
- **People**: Name, email, role
- **Comments**: Comment text, author

Show count per category in results.

### 4. Filters and Facets

Allow refining search results:
- Filter by content type (checkbox per type)
- Filter by date range (created, modified)
- Filter by owner/creator
- Filter by status (for tasks: open, closed)
- Filter by tags/labels

Filters should update results instantly.

### 5. Advanced Search Syntax

Support power-user search operators:
- `"exact phrase"` - Exact match
- `type:task` - Filter by type
- `owner:@username` - Filter by owner
- `created:>2024-01-01` - Date filters
- `is:unread` - Status filters
- `-keyword` - Exclude term
- `OR` - Boolean or (default is AND)

### 6. Search Results Page

Full results page at /search:
- All filters visible in sidebar
- Paginated results (20 per page)
- Sort options: relevance, date, name
- Save search as filter/view
- Export results to CSV

### 7. Search Analytics

Track search behavior:
- Log all search queries (anonymized)
- Track click-through rate per result
- Identify zero-result queries
- Surface trending searches

## Success Criteria

- Search results should feel instant
- Search should find relevant results
- Users should be able to find what they're looking for
- Improve user productivity

## Technical Notes

- Consider Elasticsearch or Algolia
- Index updates should be near real-time
- Handle special characters and Unicode
