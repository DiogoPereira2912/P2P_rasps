#!/bin/bash

cd "$(dirname "$0")" || exit

IPS=("192.168.0.105, 192.168.0.104, 192.168.0.103, 192.168.0.101") # substituir pelos IPs dos Raspberries

echo "📦 A guardar no GitHub..."
git add .
git commit -m "$1"
git push --force origin main

echo "🚀 A fazer deploy para os raspberries..."
for ip in "${IPS[@]}"; do 
    rsync -az --delete --exclude-from='.rsyncignore' ./ aidi@$ip:/home/aidi/P2P/ # substituir dirs dos rasps
done

echo "✅ Sucesso! Código no GitHub e nos Raspberries."