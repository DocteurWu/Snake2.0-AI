#!/bin/bash
# Lance l'entraînement 24h en mode headless.
# Important : rediriger stdout vers un fichier (et -u unbuffered) sinon le
# process dort en attendant que le pipe Hermes soit consommé.
cd /home/dietpi/Snake2.0-AI
exec .venv/bin/python -u run_24h.py >> run_24h.out 2>&1
