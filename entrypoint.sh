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

# Set up cron job only if enabled
if [ "${ENABLE_CRON:-false}" = "true" ]; then
    CRON_SCHEDULE="${COLLECTION_SCHEDULE:-0 3 * * *}"

    # Next run only: DEBUG logging for diagnostics
    DEBUG_CMD="LOG_LEVEL=DEBUG cd /app && python3 /app/run_pipeline.py --config-dir /app/config --data-dir /app/data --web-dir /app/web > /app/logs/cron.log 2>&1"
    if [ -n "${BUTTONDOWN_API_KEY:-}" ]; then
        DEBUG_CMD="${DEBUG_CMD} && python3 /app/scripts/send_newsletter.py >> /app/logs/newsletter.log 2>&1"
    fi

    # Restore normal logging for subsequent runs (after 24h)
    NORMAL_CMD="cd /app && python3 /app/run_pipeline.py --config-dir /app/config --data-dir /app/data --web-dir /app/web >> /app/logs/cron.log 2>&1"
    if [ -n "${BUTTONDOWN_API_KEY:-}" ]; then
        NORMAL_CMD="${NORMAL_CMD} && python3 /app/scripts/send_newsletter.py >> /app/logs/newsletter.log 2>&1"
    fi

    echo "$CRON_SCHEDULE $NORMAL_CMD" > /etc/cron.d/ai-news-cron-restore
    (sleep 86400 && mv /etc/cron.d/ai-news-cron-restore /etc/cron.d/ai-news-cron && crontab /etc/cron.d/ai-news-cron) &

    echo "$CRON_SCHEDULE $DEBUG_CMD" > /etc/cron.d/ai-news-cron
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