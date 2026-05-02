param(
    [string]$BundleRoot = "D:\毕业设计\LPSCl_UMA_transport_project\06_cloud_vm_gpu_bundle",
    [string]$OutputArchive = "D:\毕业设计\LPSCl_UMA_transport_project\06_cloud_vm_gpu_bundle.tar.gz"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $BundleRoot)) {
    throw "Bundle root not found: $BundleRoot"
}

$parent = Split-Path -Parent $BundleRoot
$name = Split-Path -Leaf $BundleRoot

Write-Host "[INFO] Bundle root: $BundleRoot"
Write-Host "[INFO] Output archive: $OutputArchive"

if (Test-Path $OutputArchive) {
    Remove-Item -Force $OutputArchive
}

Push-Location $parent
try {
    tar -caf $OutputArchive $name
}
finally {
    Pop-Location
}

Write-Host "[INFO] Archive created: $OutputArchive"
