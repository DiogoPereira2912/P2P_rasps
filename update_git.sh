#!/bin/bash

cd "$(dirname "$0")" || exit

IPS=("172.31.1.118") # substituir pelos IPs dos Raspberries

echo "📦 A guardar no GitHub..."
git add .
git commit -m "$1"
git push --force origin main

echo "🚀 A fazer deploy para os raspberries..."
for ip in "${IPS[@]}"; do 
    rsync -az --delete --exclude='.git' ./ aidi@$ip:/home/aidi/P2P_rasps/ # substituir dirs dos rasps
done

echo "✅ Sucesso! Código no GitHub e nos Raspberries."