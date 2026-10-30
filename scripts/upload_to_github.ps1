param(
    [string]$Archive = (Join-Path $PSScriptRoot 'machine_learning_github_ready.zip'),
    [string]$Repository = 'https://github.com/Scluzlep/machine_learning.git'
)

$ErrorActionPreference = 'Stop'
$target = Join-Path $PSScriptRoot 'machine_learning_git_upload'
if (-not (Test-Path -LiteralPath $Archive)) { throw "Archive not found: $Archive" }
if (Test-Path -LiteralPath $target) { throw "Target already exists, please remove or rename: $target" }

& git clone $Repository $target
if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }

Expand-Archive -LiteralPath $Archive -DestinationPath $target -Force
Push-Location $target
try {
    & git add -A
    if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host 'No new changes to upload.'
    } elseif ($LASTEXITCODE -eq 1) {
        & git commit -m 'Import AutoDL machine learning lessons and environment snapshot'
        if ($LASTEXITCODE -ne 0) { throw 'git commit failed. Configure git user.name and user.email if necessary.' }
        & git push origin HEAD:main
        if ($LASTEXITCODE -ne 0) { throw 'git push failed. Check GitHub authentication and repository permissions.' }
        Write-Host 'Repository upload complete.'
    } else {
        throw 'git diff failed'
    }
} finally {
    Pop-Location
}
