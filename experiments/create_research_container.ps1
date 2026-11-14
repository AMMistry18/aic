param([string]$Name = 'research-aic-eval')
$ErrorActionPreference = 'Stop'
$researchRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$image = 'ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'
docker info --format '{{.OSType}} {{.ServerVersion}}'
if ($LASTEXITCODE -ne 0) {
    Start-Process -FilePath 'C:\Program Files\Docker\Docker\Docker Desktop.exe' -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds(60)
    do {
        Start-Sleep -Seconds 2
        docker info --format '{{.OSType}} {{.ServerVersion}}' 2>$null
        if ($LASTEXITCODE -eq 0) { break }
    } while ((Get-Date) -lt $deadline)
    if ($LASTEXITCODE -ne 0) { throw 'Docker Linux engine did not become ready.' }
}
$existing = docker ps -a --filter "name=^/$Name$" --format '{{.Names}}'
if ($existing -eq $Name) {
    $actualImage = docker inspect $Name --format '{{.Config.Image}}'
    if ($actualImage -ne $image) { throw "Existing $Name uses a different image; choose another name." }
    docker start $Name | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not start $Name" }
} else {
    docker pull $image
    if ($LASTEXITCODE -ne 0) { throw 'Pinned evaluator image could not be pulled.' }
    docker run -d --name $Name --label 'aic.research=lastinch' --ipc shareable --shm-size 2g --entrypoint bash --mount "type=bind,source=$researchRoot,target=/research" $image -c 'sleep infinity'
    if ($LASTEXITCODE -ne 0) { throw 'Research container could not be created.' }
}
Write-Output "Ready: $Name. Run docker exec $Name bash /research/experiments/start_kilted_handoff.sh once."
