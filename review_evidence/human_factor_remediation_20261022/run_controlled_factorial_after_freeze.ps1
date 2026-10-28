$ErrorActionPreference = 'Stop'

$workspace = (Resolve-Path -LiteralPath 'D:\Desktop\UAV新').Path
$python = 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe'
$relativeRoot = 'review_evidence\human_factor_remediation_20261022'
$mainScript = Join-Path $workspace "$relativeRoot\controlled_factorial_diagnostic.py"
$validator = Join-Path $workspace "$relativeRoot\validate_controlled_factorial_diagnostic.py"
$spaceWriter = Join-Path $workspace "$relativeRoot\write_space_and_cleanup_record.py"
$runLog = Join-Path $workspace "$relativeRoot\run.log"
$temporary = Join-Path $workspace '.tmp\pilot_distribution_exp_20261023'

foreach ($required in @($python, $mainScript, $validator, $spaceWriter)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is missing: $required"
    }
}

Set-Location -LiteralPath $workspace
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

if (Test-Path -LiteralPath $temporary) {
    $existing = @(Get-ChildItem -LiteralPath $temporary -Force)
    if ($existing.Count -ne 0) {
        throw 'Temporary directory is not empty; preserve and inspect it before rerun.'
    }
} else {
    New-Item -ItemType Directory -Path $temporary | Out-Null
}

$env:TEMP = $temporary
$env:TMP = $temporary
$env:PYTHONDONTWRITEBYTECODE = '1'

$header = @(
    "[$([DateTimeOffset]::Now.ToString('o'))] frozen-source controlled factorial run",
    'Interpreter: D:/ProgramData/Anaconda/envs/pytorch312-lmdt/python.exe',
    'Main: review_evidence/human_factor_remediation_20261022/controlled_factorial_diagnostic.py',
    'Validator: review_evidence/human_factor_remediation_20261022/validate_controlled_factorial_diagnostic.py',
    ''
)
Set-Content -LiteralPath $runLog -Value $header -Encoding utf8

$validated = $false
try {
    & $python -m py_compile $mainScript $validator $spaceWriter
    if ($LASTEXITCODE -ne 0) {
        throw "py_compile failed with exit code $LASTEXITCODE"
    }

    & $python $mainScript 2>&1 | Tee-Object -FilePath $runLog -Append
    $mainExit = $LASTEXITCODE
    if ($mainExit -ne 0) {
        throw "controlled factorial diagnostic failed with exit code $mainExit"
    }

    & $python $validator 2>&1 | Tee-Object -FilePath $runLog -Append
    $validationExit = $LASTEXITCODE
    if ($validationExit -ne 0) {
        throw "identity/integrity validation failed with exit code $validationExit"
    }
    $validated = $true
} finally {
    if (Test-Path -LiteralPath $temporary) {
        $remaining = @(Get-ChildItem -LiteralPath $temporary -Force)
        if ($remaining.Count -eq 0) {
            Remove-Item -LiteralPath $temporary
            Add-Content -LiteralPath $runLog -Encoding utf8 -Value (
                'Temporary cleanup: removed verified empty directory; exists_after=false.'
            )
        } else {
            Add-Content -LiteralPath $runLog -Encoding utf8 -Value (
                "Temporary cleanup: preserved non-empty directory with $($remaining.Count) items."
            )
        }
    }
}

if (-not $validated) {
    throw 'Final validation did not pass; outputs remain diagnostic/intermediate.'
}

& $python $spaceWriter 2>&1 | Tee-Object -FilePath $runLog -Append
if ($LASTEXITCODE -ne 0) {
    throw "space/cleanup record generation failed with exit code $LASTEXITCODE"
}

Write-Output 'Controlled factorial diagnostic rerun completed with identity validation PASS.'
