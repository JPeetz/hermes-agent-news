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
TWITTERAPI_IO_KEY=${TWITTERAPI_IO_KEY:-}
PIPELINE_BASE_URL=${PIPELINE_BASE_URL:-https://hermesnews.xyz}
TARGET_DATE=${TARGET_DATE:-}
EOF

# Set up cron job only if enabled
if [ "${ENABLE_CRON:-false}" = "true" ]; then
    CRON_SCHEDULE="${COLLECTION_SCHEDULE:-0 3 * * *}"

    # Both run_pipeline.py and send_newsletter.py call load_dotenv() and read
    # /app/.env (written above), so cron children get their env from there.
    # NEVER bake secrets into the cron command line: a "VAR=x cmd1 && cmd2"
    # prefix only scopes the vars to cmd1 (cd), and the keys leak in `ps`.
    PIPE_CMD="cd /app && python3 /app/run_pipeline.py --config-dir /app/config --data-dir /app/data --web-dir /app/web"
    if [ -n "${BUTTONDOWN_API_KEY:-}" ]; then
        PIPE_CMD="${PIPE_CMD} && python3 /app/scripts/send_newsletter.py"
    fi

    # Normal run
    echo "PATH=/usr/local/bin:/usr/bin:/bin" > /etc/cron.d/ai-news-cron
    # Wrap the whole chain in a subshell so ALL output (run_pipeline AND
    # send_newsletter) is captured to cron.log. Without the subshell, shell
    # precedence binds ">> log 2>&1" only to the LAST command (send_newsletter),
    # silently dropping every run_pipeline log line — which hid pipeline
    # errors/tracebacks. (Fixed 2026-10-05.)
    echo "$CRON_SCHEDULE ( $PIPE_CMD ) >> /app/logs/cron.log 2>&1" >> /etc/cron.d/ai-news-cron
    chmod 0644 /etc/cron.d/ai-news-cron
    crontab /etc/cron.d/ai-news-cron
    echo "Cron job scheduled: $CRON_SCHEDULE"
    cron
else
    echo "Cron scheduler disabled (set ENABLE_CRON=true to enable)"
fi

# Start nginx in foreground
echo "Starting web server on port 80..."
nginx -g 'daemon off;'