# validate-submission.ps1 — content-moderation-openenv
# Mirrors bash validate-submission.sh logic for Windows environments

param(
    [string]$PingUrl = "https://rabedatasets-content-moderation-openenv.hf.space",
    [string]$RepoDir = "."
)

# configuration
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# color codes
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Cyan = "Cyan"

# utility functions

function Get-Timestamp {
    return Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}

function Write-Log {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    $timestamp = Get-Timestamp
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "================================" -ForegroundColor $Cyan
    Write-Host $Text -ForegroundColor $Cyan
    Write-Host "================================" -ForegroundColor $Cyan
}

function Write-Pass {
    Write-Log "✓ PASSED" -Color $Green
}

function Write-Fail {
    param([string]$Message)
    Write-Log "✗ FAILED: $Message" -Color $Red
}

function Write-Hint {
    param([string]$Message)
    Write-Log "💡 Hint: $Message" -Color $Yellow
}

# step 1: ping huggingface space

function Step-1-Ping-HF-Space {
    Write-Header "STEP 1: Ping HuggingFace Space"
    
    Write-Log "Attempting to reach: $PingUrl/reset"
    
    try {
        $uri = "$PingUrl/reset"
        $body = @{} | ConvertTo-Json
        
        $response = Invoke-RestMethod -Uri $uri `
                                     -Method Post `
                                     -Body $body `
                                     -ContentType "application/json" `
                                     -TimeoutSec 10 `
                                     -ErrorAction Stop
        
        if ($response -or $true) {
            Write-Log "HTTP 200 OK - HF Space is reachable" -Color $Green
            Write-Pass
            return $true
        }
    }
    catch [System.Net.Http.HttpRequestException] {
        Write-Fail "HTTP request failed: Connection or HTTP error"
        Write-Hint "Make sure the HF Space URL is correct: $PingUrl"
        Write-Hint "Verify the space is deployed and accessible"
        return $false
    }
    catch [System.Net.WebException] {
        Write-Fail "Connection error: Cannot reach the server"
        Write-Hint "Check internet connection"
        Write-Hint "Verify HF_TOKEN if space is private"
        return $false
    }
    catch {
        Write-Fail "Unexpected error: $($_.Exception.Message)"
        Write-Hint "Check the URL format: https://username-project.hf.space"
        return $false
    }
}

# step 2: docker build

function Step-2-Docker-Build {
    Write-Header "STEP 2: Docker Build"
    
    # check if docker is installed
    Write-Log "Checking if Docker is installed..."
    
    try {
        $docker = Get-Command docker -ErrorAction Stop
        Write-Log "Docker found at: $($docker.Source)" -Color $Green
    }
    catch {
        Write-Fail "Docker command not found"
        Write-Hint "Install Docker from: https://www.docker.com/products/docker-desktop"
        return $false
    }
    
    # check if Dockerfile exists
    Write-Log "Checking for Dockerfile..."
    $dockerfilePath = Join-Path $RepoDir "Dockerfile"
    
    if (-not (Test-Path $dockerfilePath)) {
        Write-Fail "Dockerfile not found at: $dockerfilePath"
        Write-Hint "Make sure Dockerfile exists in the repo root"
        return $false
    }
    
    Write-Log "Dockerfile found at: $dockerfilePath" -Color $Green
    
    # run docker build
    Write-Log "Running: docker build ."
    
    try {
        Push-Location $RepoDir
        
        # capture output
        $output = & docker build . 2>&1
        $exitCode = $LASTEXITCODE
        
        Pop-Location
        
        if ($exitCode -eq 0) {
            Write-Log "Docker build completed successfully" -Color $Green
            Write-Pass
            return $true
        }
        else {
            Write-Fail "Docker build failed with exit code: $exitCode"
            Write-Log "Last 20 lines of output:" -Color $Red
            
            # print last 20 lines
            $lines = $output -split "`n"
            $lastLines = $lines | Select-Object -Last 20
            foreach ($line in $lastLines) {
                if ($line.Trim()) {
                    Write-Host "  $line" -ForegroundColor $Red
                }
            }
            
            return $false
        }
    }
    catch {
        Write-Fail "Docker build command failed: $($_.Exception.Message)"
        return $false
    }
}

# step 3: openenv validate

function Step-3-OpenEnv-Validate {
    Write-Header "STEP 3: OpenEnv Validate"
    
    # check if openenv command exists
    Write-Log "Checking if openenv is installed..."
    
    try {
        $openenv = Get-Command openenv -ErrorAction Stop
        Write-Log "openenv found at: $($openenv.Source)" -Color $Green
    }
    catch {
        Write-Fail "openenv command not found"
        Write-Hint "Install openenv-core with: pip install openenv-core"
        return $false
    }
    
    # run openenv validate
    Write-Log "Running: openenv validate"
    
    try {
        Push-Location $RepoDir
        
        # capture output
        $output = & openenv validate 2>&1
        $exitCode = $LASTEXITCODE
        
        Pop-Location
        
        if ($exitCode -eq 0) {
            Write-Log "openenv validation passed" -Color $Green
            Write-Pass
            return $true
        }
        else {
            Write-Fail "openenv validation failed with exit code: $exitCode"
            Write-Log "Validation output:" -Color $Red
            
            # print all output
            foreach ($line in $output) {
                if ($line.Trim()) {
                    Write-Host "  $line" -ForegroundColor $Red
                }
            }
            
            return $false
        }
    }
    catch {
        Write-Fail "openenv command failed: $($_.Exception.Message)"
        return $false
    }
}

# main execution

function Main {
    Write-Header "Content Moderation OpenEnv — Hackathon Validation"
    
    Write-Log "Starting validation..." -Color $Cyan
    Write-Log "Ping URL: $PingUrl" -Color $Cyan
    Write-Log "Repo Dir: $RepoDir" -Color $Cyan
    
    # run all steps
    $step1Pass = Step-1-Ping-HF-Space
    $step2Pass = Step-2-Docker-Build
    $step3Pass = Step-3-OpenEnv-Validate
    
    # summary
    Write-Header "SUMMARY"
    
    $passCount = @($step1Pass, $step2Pass, $step3Pass) | Where-Object { $_ } | Measure-Object | Select-Object -ExpandProperty Count
    $totalCount = 3
    
    if ($passCount -eq $totalCount) {
        Write-Log "All $totalCount/$totalCount checks passed! Ready to submit." -Color $Green
        Write-Host ""
        exit 0
    }
    else {
        Write-Log "$passCount/$totalCount checks passed" -Color $Red
        Write-Log "Fix the failures above and try again" -Color $Red
        Write-Host ""
        exit 1
    }
}

# run main
Main

