param(
    [Parameter(Mandatory=$true)]
    [string]$InputLas,

    [Parameter(Mandatory=$true)]
    [string]$OutputDir
)

$ErrorActionPreference = "Continue"

if (-not (Test-Path $InputLas)) {
    throw "Input LAS/LAZ not found: $InputLas"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Results = @()

function Run-P2SBL {
    param(
        [string]$Name,
        [string[]]$CliArgs
    )

    $Output = Join-Path $OutputDir ($Name + ".las")
    Write-Host ""
    Write-Host "================================================================"
    Write-Host "[RUN] $Name"
    Write-Host "[OUT] $Output"
    Write-Host "================================================================"

    $FullArgs = @(
        "predict",
        "--in_las", $InputLas,
        "--out_las", $Output
    ) + $CliArgs

    $Watch = [System.Diagnostics.Stopwatch]::StartNew()
    & points2sbl @FullArgs
    $ExitCode = $LASTEXITCODE
    $Watch.Stop()

    $Success = ($ExitCode -eq 0 -and (Test-Path $Output))

    $script:Results += [PSCustomObject]@{
        Run      = $Name
        Success  = $Success
        ExitCode = $ExitCode
        Minutes  = [math]::Round($Watch.Elapsed.TotalMinutes, 2)
        Output   = $Output
    }

    if ($Success) {
        Write-Host "[PASS] $Name"
    } else {
        Write-Warning "FAILED: $Name"
    }
}

# Naive-user interfaces.
Run-P2SBL "01_NAIVE_RAW" @("--mode","raw")
Run-P2SBL "02_NAIVE_FULL" @("--mode","full")
Run-P2SBL "03_NAIVE_ADAPTIVE" @("--mode","adaptive")

# Raw threshold / vote-layout tests.
Run-P2SBL "04_RAW_1VOTE_THR050" @("--mode","raw","--votes","1","--thr","0.50")
Run-P2SBL "05_RAW_1VOTE_THR090" @("--mode","raw","--votes","1","--thr","0.90")
Run-P2SBL "06_RAW_GRID4_4VOTES" @("--mode","raw","--votes","4","--vote_mode","grid4","--vote_weight","confidence")
Run-P2SBL "07_RAW_GRID8_8VOTES" @("--mode","raw","--votes","8","--vote_mode","grid8","--vote_weight","confidence")
Run-P2SBL "08_RAW_HYBRID8_8VOTES" @("--mode","raw","--votes","8","--vote_mode","hybrid8","--vote_weight","confidence","--seed","42")
Run-P2SBL "09_RAW_RANDOM_8VOTES" @("--mode","raw","--votes","8","--vote_mode","random","--vote_weight","confidence","--seed","42")
Run-P2SBL "10_RAW_RANDOM_12VOTES" @("--mode","raw","--votes","12","--vote_mode","random","--vote_weight","confidence","--seed","42")

# Full pipeline.
Run-P2SBL "11_FULL_DEFAULT" @("--mode","full")
Run-P2SBL "12_FULL_GRID4_4VOTES" @("--mode","full","--votes","4","--vote_mode","grid4")
Run-P2SBL "13_FULL_GRID8_8VOTES" @("--mode","full","--votes","8","--vote_mode","grid8")
Run-P2SBL "14_FULL_HYBRID8_8VOTES" @("--mode","full","--votes","8","--vote_mode","hybrid8","--seed","42")

# Adaptive probability-distribution mode.
Run-P2SBL "15_ADAPTIVE_DEFAULT" @("--mode","adaptive")
Run-P2SBL "16_ADAPTIVE_GRID8" @("--mode","adaptive","--votes","8","--vote_mode","grid8")
Run-P2SBL "17_ADAPTIVE_HYBRID8" @("--mode","adaptive","--votes","8","--vote_mode","hybrid8","--seed","42")
Run-P2SBL "18_ADAPTIVE_RANDOM8" @("--mode","adaptive","--votes","8","--vote_mode","random","--seed","42")
Run-P2SBL "19_ADAPTIVE_RANDOM12" @("--mode","adaptive","--votes","12","--vote_mode","random","--seed","42")
Run-P2SBL "20_ADAPTIVE_HYBRID8_SHOULDER001" @("--mode","adaptive","--votes","8","--vote_mode","hybrid8","--seed","42","--adaptive_shoulder_fraction","0.01")

$CSV = Join-Path $OutputDir "points2sbl_test_matrix_summary.csv"
$Results | Export-Csv -Path $CSV -NoTypeInformation
$Results | Format-Table Run, Success, ExitCode, Minutes -AutoSize

Write-Host ""
Write-Host "[SUMMARY] $CSV"
