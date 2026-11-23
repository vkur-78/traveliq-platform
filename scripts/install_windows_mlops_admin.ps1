<#
.SYNOPSIS
    Installs WSL2 and Docker Desktop (official sources) on Windows 11.
    Requires Administrator (UAC) approval and a system reboot after WSL2 enablement.
#>

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
    Write-Error "Please run this script from an Administrator PowerShell prompt (Windows UAC approval required)."
    exit 1
}

Write-Host "=== 1. Installing / Enabling WSL2 (Official Microsoft) ===" -ForegroundColor Cyan
wsl --install --no-distribution
wsl --set-default-version 2 2>$null

Write-Host "=== 2. Installing Docker Desktop (Official winget package: Docker.DockerDesktop) ===" -ForegroundColor Cyan
winget install --id Docker.DockerDesktop -e --source winget --accept-package-agreements --accept-source-agreements

Write-Host "`n=== NEXT STEPS ===" -ForegroundColor Green
Write-Host "1. Restart Windows so the WSL2 kernel & Virtual Machine Platform finish activating."
Write-Host "2. After reboot, launch Docker Desktop, wait for Engine running, and optionally enable Kubernetes in Settings -> Kubernetes."
Write-Host "3. Return to Antigravity and say 'Docker is running' so we can execute Docker Compose, Kubernetes + HPA, and Airflow live."
