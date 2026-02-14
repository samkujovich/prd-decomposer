# PRD: Notification Center

## Overview

Build a centralized notification system to keep users informed about important events across the platform.

## Background

Users currently miss important updates because:
- Email-only notifications get lost
- No in-app notification history
- Can't customize what they receive

## Requirements

### 1. Notification Bell

Add notification bell icon to the header:
- Show unread count badge (max "99+")
- Clicking opens notification dropdown
- Bell should animate briefly when new notification arrives
- Position: right side of header, before user avatar

### 2. Notification Dropdown

When clicking the bell, show dropdown with:
- List of recent notifications (last 20)
- Each notification shows: icon, title, timestamp, read/unread state
- "Mark all as read" link at top
- "View all" link at bottom
- Clicking a notification navigates to relevant page

### 3. Full Notification Page

Create /notifications page showing:
- All notifications with pagination (20 per page)
- Filter by type (mentions, assignments, updates, system)
- Filter by read/unread status
- Bulk actions: mark selected as read, delete selected
- Search notifications by keyword

### 4. Notification Types

Support these notification categories:
- **Mentions**: When someone @mentions the user
- **Assignments**: When assigned to a task/project
- **Updates**: Changes to items user is watching
- **Comments**: Replies to user's comments
- **System**: Account alerts, billing, maintenance

Each type should have a distinct icon.

### 5. Real-time Delivery

Notifications should appear instantly:
- Use WebSocket connection for real-time push
- Fall back to polling if WebSocket fails
- Show toast notification for high-priority items
- Play subtle sound (user can disable)

### 6. Notification Preferences

Users can customize notifications at /settings/notifications:
- Toggle each notification type on/off
- Choose delivery method per type: in-app, email, both, none
- Set quiet hours (no push during specified times)
- Email digest option: immediate, daily, weekly, never

### 7. Mobile Support

Notifications should work well on mobile:
- Responsive dropdown design
- Touch-friendly tap targets
- Swipe to dismiss on mobile

## Success Criteria

- Notifications delivered within 2 seconds
- Users should engage more with the notification system
- Reduce email notification volume by 40%
