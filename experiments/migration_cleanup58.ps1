$ErrorActionPreference = 'Stop'
$researchRoot = [IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$exportRoot = [IO.Path]::GetFullPath((Join-Path (Split-Path $researchRoot -Parent) 'migration-export-20261114'))
$statusPath = Join-Path $exportRoot 'cleanup-progress.json'
$state = @{ status = 'waiting_for_verified_backup'; removed = @(); failures = @(); started = (Get-Date).ToString('o') }
function Save-State { $state | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $statusPath -Encoding UTF8 }
function Resolve-ScopedPath([string]$Path, [string]$AllowedRoot) {
    $absolute = [IO.Path]::GetFullPath($Path)
    $allowed = [IO.Path]::GetFullPath($AllowedRoot)
    if ($absolute -ne $allowed -and -not $absolute.StartsWith($allowed + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Deletion target escaped approved project path: $absolute"
    }
    return $absolute
}
Save-State
try {
    while ($true) {
        $uploadPath = Join-Path $exportRoot 'upload-progress.json'
        $verifyPath = Join-Path $exportRoot 'archive-verification.json'
        if ((Test-Path -LiteralPath $uploadPath) -and (Test-Path -LiteralPath $verifyPath)) {
            $upload = Get-Content -LiteralPath $uploadPath -Raw | ConvertFrom-Json
            $verify = Get-Content -LiteralPath $verifyPath -Raw | ConvertFrom-Json
            if ($upload.status -eq 'stopped_requires_inspection' -or $verify.status -eq 'stopped_requires_inspection') {
                throw 'Backup/upload verification stopped. No cleanup authorized by this worker.'
            }
            if ($upload.status -eq 'complete_remote_verified' -and $verify.status -eq 'complete_verified') { break }
        }
        Start-Sleep -Seconds 15
    }
    $head = (& git -C $researchRoot rev-parse HEAD).Trim()
    $remote = (& git -C $researchRoot ls-remote --heads origin research/gazebo-rap-evaluation)
    if ($LASTEXITCODE -ne 0 -or -not $remote.StartsWith($head)) { throw 'Latest code is not verified on remote branch.' }
    $state.status = 'removing_only_verified_project_data'
    $state.free_bytes_before = ([IO.DriveInfo]::new([IO.Path]::GetPathRoot($researchRoot))).AvailableFreeSpace
    Save-State
    # Require both saved runtime images and every archive part to have a
    # matching remote verification record before deleting containers or data.
    foreach ($indexName in @('workspace-index.json', 'legacy-index.json', 'docker-index.json')) {
        $index = Get-Content -LiteralPath (Join-Path $exportRoot $indexName) -Raw | ConvertFrom-Json
        foreach ($part in $index.parts) {
            $remotePart = @($upload.verified_assets | Where-Object { $_.name -eq $part.name -and $_.sha256 -eq $part.sha256 -and $_.size -eq $part.size })
            if ($remotePart.Count -ne 1) { throw "Unverified remote part: $($part.name)" }
        }
    }
    foreach ($name in @('research-aic-eval', 'research-aic-perception30', 'research-aic-train')) {
        $raw = & docker inspect $name 2>$null
        if ($LASTEXITCODE -ne 0) { continue }
        $container = @($raw | ConvertFrom-Json)[0]
        if ($container.State.Running) { throw "Research container unexpectedly running: $name" }
        $owned = @($container.Mounts | Where-Object { $_.Destination -eq '/research' -and [IO.Path]::GetFullPath($_.Source) -eq $researchRoot })
        if ($owned.Count -ne 1) { throw "Research mount identity not established: $name" }
        & docker rm $container.Id
        if ($LASTEXITCODE -ne 0) { throw "Failed to remove owned stopped container: $name" }
        $state.removed += "container:$name"; Save-State
    }
    foreach ($tag in @('aic-research-migration:baseline-20261114', 'aic-research-migration:development-20261114',
        'flowstate:aic_model', 'aic_perception:check_board_visibility_skill_v4', 'ghcr.io/intrinsic-dev/aic/aic_eval:latest')) {
        & docker image inspect $tag *> $null
        if ($LASTEXITCODE -ne 0) { continue }
        & docker image rm $tag
        if ($LASTEXITCODE -eq 0) { $state.removed += "image:$tag" }
        else { $state.failures += "Image retained because Docker refused removal: $tag" }
        Save-State
    }
    # These two absolute paths are explicitly within the Intrinsic workspace.
    # Their sources/models/raw evidence are remotely verified. Older Flowstate
    # image-bundle tar files are obsolete generated builds, intentionally omitted.
    $artifactRoot = Resolve-ScopedPath (Join-Path $researchRoot '.artifacts') $researchRoot
    $oldBuildRoot = [IO.Path]::GetFullPath((Join-Path (Split-Path $researchRoot -Parent) '.flowstate-build'))
    $intrinsicWorkspace = [IO.Path]::GetFullPath((Split-Path $researchRoot -Parent))
    $oldBuildRoot = Resolve-ScopedPath $oldBuildRoot $intrinsicWorkspace
    foreach ($target in @($artifactRoot, $oldBuildRoot)) {
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
            $state.removed += $target; Save-State
        }
    }
    $tracked = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    & git -C $researchRoot -c core.quotepath=false ls-files | ForEach-Object { [void]$tracked.Add($_) }
    $index = Get-Content -LiteralPath (Join-Path $exportRoot 'workspace-index.json') -Raw | ConvertFrom-Json
    $removedFiles = 0; $retainedChanged = @()
    foreach ($row in $index.files) {
        if ($row.path.StartsWith('.artifacts/') -or $tracked.Contains($row.path)) { continue }
        $target = Resolve-ScopedPath (Join-Path $researchRoot $row.path) $researchRoot
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { continue }
        $digest = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -ne $row.sha256) { $retainedChanged += $row.path; continue }
        Remove-Item -LiteralPath $target -Force
        $removedFiles++
    }
    $state.removed_archived_untracked_files = $removedFiles
    $state.retained_changed_files = $retainedChanged
    # Delete only verified local upload parts, keeping manifests and receipts.
    foreach ($asset in $upload.verified_assets) {
        if ($asset.name -notmatch '\.tar\.gz\.part[0-9]+$') { continue }
        $target = Resolve-ScopedPath (Join-Path $exportRoot $asset.name) $exportRoot
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Force }
    }
    $state.free_bytes_after = ([IO.DriveInfo]::new([IO.Path]::GetPathRoot($researchRoot))).AvailableFreeSpace
    $state.freed_host_bytes = $state.free_bytes_after - $state.free_bytes_before
    $state.status = 'completed_verified_scoped_cleanup'
    $state.ended = (Get-Date).ToString('o')
    Save-State
} catch {
    $state.status = 'stopped_requires_inspection'; $state.error = $_.Exception.Message; Save-State
    throw
}
