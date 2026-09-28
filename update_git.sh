#!/bin/bash

# This script updates the local git repository by committing any local changes


git add .
git commit -m "$1"
git push --force origin main