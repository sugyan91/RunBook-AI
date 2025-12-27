cat > run_gui.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONPATH="$(pwd)"
python -m streamlit run app/gui.py
EOF
chmod +x run_gui.sh

