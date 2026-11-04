# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY frontend/ ./

# Build the frontend (outputs to ../web)
RUN npm run build

# Generate the crawler-facing shell: identical SPA, but with the GEO SSI includes
# injected after <body> so nginx can bake the quick-answer + edition-seo blocks
# into the raw HTML served to AI-search crawlers ($is_crawler swap in nginx.conf).
# Humans get the clean index.html; only crawlers get this file.
# NOTE: this stage is node:20-alpine — no python3, so use node for the transform.
RUN node - <<'EOF'
const fs = require('fs');
const html = fs.readFileSync('../web/index.html', 'utf8');
const inject = '<!--#include virtual="/data/quick-answer.html" -->\n<!--#include virtual="/data/edition-seo.html" -->\n';
const m = html.match(/<body[^>]*>/);
if (!m) throw new Error('body tag not found');
const out = html.slice(0, m.index + m[0].length) + inject + html.slice(m.index + m[0].length);
fs.writeFileSync('../web/index.crawler.html', out);
console.log('wrote index.crawler.html', out.length, 'bytes');
EOF

# Stage 2: Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cron \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser runtime used by the LessWrong cookie bypass
RUN python -m playwright install --with-deps chromium

# Copy application code
COPY agents/ ./agents/
COPY shadow/ ./shadow/
COPY config/shadow/news-relevance-v5-frozen.json ./config/shadow/news-relevance-v5-frozen.json
COPY generators/ ./generators/
COPY scripts/ ./scripts/
COPY assets/ ./assets/
COPY run_pipeline.py .
COPY entrypoint.sh .

# Create necessary directories
RUN mkdir -p /app/config /app/data /app/web /app/logs

# Copy built frontend from stage 1
COPY --from=frontend-builder /frontend/../web ./web/

# Make scripts executable
RUN chmod +x run_pipeline.py entrypoint.sh

# Configure nginx
COPY nginx.conf /etc/nginx/sites-available/default
# Anti-scraping AI block (http-level config, loaded by conf.d/)
COPY ai-block.conf /etc/nginx/conf.d/ai-block.conf

# Expose web port
EXPOSE 80

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
