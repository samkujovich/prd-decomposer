# PRD: User Onboarding Flow

## Overview

Create a guided onboarding experience for new users that improves activation rates and helps users discover key product features.

## Background

Current state:
- 40% of new signups never complete their profile
- Users report confusion about where to start
- No personalization based on user type or goals

## Requirements

### 1. Welcome Screen

After email verification, show a welcome screen with:
- Personalized greeting using the user's name
- Brief product value proposition (3-4 bullet points)
- "Get Started" CTA button
- Option to skip onboarding

### 2. User Type Selection

Allow users to identify themselves:
- **Individual**: Personal use, single account
- **Team Lead**: Will invite team members
- **Enterprise Admin**: Requires SSO/admin features

Store selection in user profile for future personalization.

### 3. Goal Setting

Based on user type, present 3-5 relevant goals:
- Individual: "Track personal tasks", "Manage projects", "Set reminders"
- Team Lead: "Collaborate with team", "Track team progress", "Assign tasks"
- Enterprise: "Manage departments", "View analytics", "Configure permissions"

Users can select multiple goals. Minimum 1 required.

### 4. Profile Completion

Collect essential profile information:
- Display name (required, 2-50 characters)
- Profile photo (optional, max 5MB, jpg/png only)
- Timezone (auto-detected, user can override)
- Notification preferences (email digest frequency)

### 5. Interactive Tutorial

After profile setup, launch a 5-step interactive tutorial:
1. Create your first project
2. Add a task
3. Set a due date
4. Invite a collaborator (skip if Individual)
5. Complete the task

Each step should have:
- Highlighted UI element with tooltip
- "Next" and "Skip" buttons
- Progress indicator (step X of 5)

### 6. Completion Celebration

On completing onboarding:
- Show confetti animation
- Display "You're all set!" message
- Provide quick links to common actions
- Offer link to help center

## Success Criteria

- Onboarding completion rate > 70%
- Time to complete < 5 minutes
- User satisfaction score > 4.0/5.0

## Technical Notes

- Store onboarding state in user record (not_started, in_progress, completed, skipped)
- Track analytics events for each step
- Support resume if user abandons mid-flow
