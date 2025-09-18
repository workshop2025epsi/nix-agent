#!/bin/bash
set -e

sudo apt-get update -y && sudo apt-get upgrade -y && sudo apt-get full-upgrade -y && sudo apt install python3 python3-dev python3-venv git -y

cd ~
mkdir -p project
cd project

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install sounddevice numpy scipy

USER_HOME=$(eval echo ~$USER)

sudo tee /etc/systemd/system/nix-agent.service > /dev/null <<EOF
[Unit]
Description=Autostart NIX Agent
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$USER_HOME/project
ExecStart=$USER_HOME/project/venv/bin/python $USER_HOME/project/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nix-agent.service
sudo systemctl start nix-agent.service

echo "Vérification du statut du service..."
sudo systemctl status nix-agent.service --no-pager
