$ErrorActionPreference = "Stop"

$ExpectedCommit = "6ea3b8d778d047b4b3b7c5b843e21c5bea98ee8d"
$SubmodulePath = "third_party/heretic"

Write-Host "Initializing pinned Heretic submodule..."
git submodule update --init --recursive -- $SubmodulePath
if ($LASTEXITCODE -ne 0) {
    throw "git submodule update failed"
}

$ActualCommit = (git -C $SubmodulePath rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read Heretic submodule commit"
}

if ($ActualCommit -ne $ExpectedCommit) {
    throw "Heretic commit mismatch. Expected $ExpectedCommit, got $ActualCommit"
}

Write-Host "Heretic pin verified: $ActualCommit"
Write-Host "Upstream environment requirements are defined inside third_party/heretic."
