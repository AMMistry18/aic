# Local laptop cleanup after renewed user instruction and complete remote backup.
# No recursive deletion: each archived file is hash-checked before removal.
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\anshu\College\Intrinsic\aic\aic'
$parentRoot = 'C:\Users\anshu\College\Intrinsic\aic'
$exportRoot = Join-Path $parentRoot 'migration-export-20261114'
if ([IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent)) -ne $root) { throw 'This cleanup is restricted to the original laptop checkout.' }
$receipt = Get-Content -LiteralPath (Join-Path $exportRoot 'upload-progress.json') -Raw | ConvertFrom-Json
if ($receipt.status -ne 'complete_remote_verified') { throw 'Backup incomplete' }
$state = @{status='deleting_individually_verified_local_duplicates'; files_removed=0; logical_bytes_removed=0L;
    skipped_changed=@(); started=(Get-Date).ToString('o'); free_before=([IO.DriveInfo]::new('C:\')).AvailableFreeSpace}
$statePath = Join-Path $exportRoot 'file-cleanup59.json'
function Save-State { $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8 }
function Scoped-Path([string]$base,[string]$relative) {
    $target=[IO.Path]::GetFullPath((Join-Path $base $relative))
    if (-not $target.StartsWith($base + '\',[StringComparison]::OrdinalIgnoreCase)) { throw "Out of scope: $target" }
    return $target
}
function Remove-VerifiedFile([string]$target,[string]$digest,[long]$bytes) {
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { return }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
        $state.skipped_changed += $target; return
    }
    Remove-Item -LiteralPath $target -Force
    $state.files_removed++; $state.logical_bytes_removed += $bytes
    if ($state.files_removed % 1000 -eq 0) { Save-State }
}
Save-State
try {
    $tracked=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    & git -C $root -c core.quotepath=false ls-files | ForEach-Object { [void]$tracked.Add($_) }
    $index=Get-Content -LiteralPath (Join-Path $exportRoot 'workspace-index.json') -Raw | ConvertFrom-Json
    foreach ($row in $index.files) {
        if ($tracked.Contains($row.path)) { continue }
        $target=Scoped-Path $root $row.path
        Remove-VerifiedFile $target $row.sha256 $row.size
    }
    $state.status='deleting_individually_verified_legacy_duplicates'; Save-State
    $legacyRoot=Join-Path $parentRoot '.flowstate-build'
    $legacy=Get-Content -LiteralPath (Join-Path $exportRoot 'legacy-index.json') -Raw | ConvertFrom-Json
    foreach ($row in $legacy.files) {
        $target=Scoped-Path $legacyRoot $row.original_relative
        Remove-VerifiedFile $target $row.sha256 $row.size
    }
    # Automatic approval review separately rejected removal of the excluded
    # old image bundles. Leave those unarchived files untouched; do not retry
    # that action indirectly through this previously running worker.
    $state.obsolete_bundles_removed=0
    $state.obsolete_bundle_deletion='blocked_by_automatic_review; excluded bundles retained'
    $state.status='completed_individual_file_cleanup'; $state.ended=(Get-Date).ToString('o')
    $state.free_after=([IO.DriveInfo]::new('C:\')).AvailableFreeSpace
    $state.freed_host_bytes=$state.free_after-$state.free_before
    Save-State
} catch {
    $state.status='stopped_requires_inspection'; $state.error=$_.Exception.Message; Save-State
    throw
}
