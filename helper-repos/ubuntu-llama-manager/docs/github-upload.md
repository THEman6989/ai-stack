# GitHub Upload

Dieses Projekt soll als eigenes Repository veröffentlicht werden, nicht im
AI-Stack-Repository.

## Lokal vorbereiten

```bash
cd /home/amin/experi/ubuntu-llama-manager
git init
git status
git add .
git commit -m "Improve manager APIs, reboot timer, and ESP integration"
```

`ubuntu-llama.conf`, `.env`, `logs/` und `state/` sind ignoriert.

## Remote erstellen

Mit GitHub CLI:

```bash
gh repo create ubuntu-llama-manager --private --source=. --remote=origin --push
```

Oder falls das Repo schon existiert:

```bash
git remote add origin git@github.com:<USER>/ubuntu-llama-manager.git
git branch -M main
git push -u origin main
```

## Vor Push prüfen

```bash
git status --short
git log --oneline -1
```
