#!/bin/bash
set -e

echo "=== Step 1: Remove .gitattributes and LFS tracking ==="
rm -f .gitattributes
git rm --cached -r . || true
git add .
git commit -m "Remove LFS tracking" || true

echo "=== Step 2: Delete all 'faster_rcnn' folders from Git history (BFG) ==="

if [ ! -f bfg.jar ]; then
    echo "Downloading BFG..."
    wget -q https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar -O bfg.jar
fi

# Correct usage: only folder name, no path
java -jar bfg.jar --delete-folders faster_rcnn --no-blob-protection

echo "=== Step 3: Cleanup Git garbage ==="
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "=== Step 4: Force push to origin/main ==="
git push origin main --force

echo "=== DONE ==="