$ErrorActionPreference = "Stop"
$env:GIT_TERMINAL_PROMPT = "0"
$remote = "https://github.com/MohamedAwadMoneer/ai-clinical-decision-support.git"
$commitMessage = "feat(core): initial release of AI Clinical Decision Support Lite RAG engine v1.0.0"

if (-not (Test-Path .git)) {
    throw "Run this script from the repository root."
}

# Use Git Credential Manager for stored HTTPS credentials without prompting Git.
git config credential.helper manager

git remote set-url origin $remote
git branch -M main

# .env is ignored; this also removes it if an old index entry exists.
git rm --cached --ignore-unmatch .env | Out-Null
git add -A
$stagedEnv = @(git diff --cached --name-only | Where-Object { $_ -eq ".env" })
if ($stagedEnv.Count -gt 0) {
    throw ".env would be committed. Stop and inspect the index."
}
git diff --cached --check

if (-not (git rev-parse --verify HEAD 2>$null)) {
    git commit -m $commitMessage
} else {
    $lastMessage = git log -1 --format=%s
    if ($lastMessage -ne $commitMessage) {
        git commit -m $commitMessage
    }
}

if (-not (git tag --list v1.0.0)) {
    git tag -a v1.0.0 -m "Release v1.0.0"
}

git push --set-upstream origin main
git push origin v1.0.0
Write-Host "Published main and v1.0.0 to $remote"
