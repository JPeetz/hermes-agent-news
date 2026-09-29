#!/bin/bash
set -e

echo "Starting AI News Aggregator..."

# Create default config files if they don't exist
if [ ! -f /app/config/rss_feeds.txt ]; then
    echo "Creating default configuration files..."
    python3 /app/run_pipeline.py --create-config --config-dir /app/config
fi

# Start newsletter subscribe server (background, port 8080)
if [ -n "${BUTTONDOWN_API_KEY:-}" ]; then
    echo "Starting newsletter subscribe server on port 8080..."
    python3 /app/scripts/subscribe_server.py &
else
    echo "BUTTONDOWN_API_KEY not set; subscribe server disabled"
fi

# Write .env file so cron children (which lose docker-compose env vars) can
# reload them via python-dotenv — run_pipeline.py already calls load_dotenv().
cat > /app/.env << EOF
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
KIE_API_KEY=${KIE_API_KEY:-}
BUTTONDOWN_API_KEY=${BUTTONDOWN_API_KEY:-}
PIPELINE_BASE_URL=${PIPELINE_BASE_URL:-https://hermesnews.xyz}
TARGET_DATE=${TARGET_DATE:-}
EOF

# Set up cron job only if enabled
if [ "${ENABLE_CRON:-false}" = "true" ]; then
    CRON_SCHEDULE="${COLLECTION_SCHEDULE:-0 3 * * *}"

    # Export essential env vars on the cron command line too, for scripts that
    # don't load dotenv (send_newsletter.py).  Quotes inside crontab are tricky
    # so use no quotes on values that lack special chars.
    VARS="LOG_LEVEL=DEBUG OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-} KIE_API_KEY=${KIE_API_KEY:-}"
    if [ -n "${BUTTONDOWN_API_KEY:-}" ]; then
        VARS="${VARS} BUTTONDOWN_API_KEY=${BUTTONDOWN_API_KEY}"
    fi

    PIPE_CMD="cd /app && python3 /app/run_pipeline.py --config-dir /app/config --data-dir /app/data --web-dir /app/web"
    if [ -n "${BUTTONDOWN_API_KEY:-}" ]; then
        PIPE_CMD="${PIPE_CMD} && python3 /app/scripts/send_newsletter.py"
    fi

    # First run (restored after 24h): DEBUG logging
    echo "PATH=/usr/local/bin:/usr/bin:/bin" > /etc/cron.d/ai-news-cron-restore
    echo "$CRON_SCHEDULE $VARS $PIPE_CMD > /app/logs/cron.log 2>&1" >> /etc/cron.d/ai-news-cron-restore
    (sleep 86400 && mv /etc/cron.d/ai-news-cron-restore /etc/cron.d/ai-news-cron && crontab /etc/cron.d/ai-news-cron) &

    # Normal run
    echo "PATH=/usr/local/bin:/usr/bin:/bin" > /etc/cron.d/ai-news-cron
    echo "$CRON_SCHEDULE $VARS $PIPE_CMD >> /app/logs/cron.log 2>&1" >> /etc/cron.d/ai-news-cron
    chmod 0644 /etc/cron.d/ai-news-cron
    crontab /etc/cron.d/ai-news-cron
    echo "Cron job scheduled (NEXT RUN DEBUG): $CRON_SCHEDULE"
    cron
else
    echo "Cron scheduler disabled (set ENABLE_CRON=true to enable)"
fi

# Start nginx in foreground
echo "Starting web server on port 80..."
nginx -g 'daemon off;'