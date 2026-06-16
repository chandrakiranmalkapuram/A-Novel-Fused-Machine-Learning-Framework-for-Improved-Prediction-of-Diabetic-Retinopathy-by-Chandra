#!/bin/bash

# Daily Git Commit and Push Script
# This script automatically commits and pushes changes to GitHub

PROJECT_DIR="/Users/chandrakiran/Documents/Dissertation/dr_project"
cd "$PROJECT_DIR" || exit 1

# Check if there are any changes
if [ -z "$(git status --short)" ]; then
    echo "[$(date)] No changes to commit" >> "$PROJECT_DIR/.git-push.log"
    exit 0
fi

# Add all changes
git add .

# Commit with timestamp
COMMIT_MESSAGE="Daily update: $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$COMMIT_MESSAGE"

# Push to origin main
git push origin main

# Log the result
if [ $? -eq 0 ]; then
    echo "[$(date)] Successfully pushed to GitHub" >> "$PROJECT_DIR/.git-push.log"
else
    echo "[$(date)] Failed to push to GitHub" >> "$PROJECT_DIR/.git-push.log"
fi
