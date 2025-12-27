cat > run_cli.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate
python -m app.main
EOF
chmod +x run_cli.sh

