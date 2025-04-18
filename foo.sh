#!/usr/bin/bash

shopt -s nullglob  # Skip if no .mmd files exist
for file in *.mmd; do
    echo "$file"
    npx -p @mermaid-js/mermaid-cli aa-exec --profile=chrome mmdc -i "$file" -o "${file%.mmd}.png"
done
