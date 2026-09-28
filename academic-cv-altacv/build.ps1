$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
$BuildDir = Join-Path $ProjectDir "build"
$XeLaTeX = "D:\texlive\2026\bin\windows\xelatex.exe"

if (-not (Test-Path -LiteralPath $XeLaTeX)) {
  $XeLaTeX = (Get-Command xelatex -ErrorAction Stop).Source
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$BuildDirForTeX = $BuildDir.Replace("\", "/")

Push-Location $ProjectDir
try {
  & $XeLaTeX "-output-directory=$BuildDirForTeX" -no-pdf -interaction=batchmode -halt-on-error -file-line-error cv.tex
  if ($LASTEXITCODE -ne 0) { throw "XeLaTeX metadata pass failed" }
  & $XeLaTeX "-output-directory=$BuildDirForTeX" -interaction=batchmode -halt-on-error -file-line-error cv.tex
  if ($LASTEXITCODE -ne 0) { throw "XeLaTeX PDF pass failed" }
} finally {
  Pop-Location
}

Copy-Item -LiteralPath (Join-Path $BuildDir "cv.pdf") -Destination (Join-Path $ProjectDir "马睿骁-学术简历.pdf") -Force
Write-Output (Join-Path $ProjectDir "马睿骁-学术简历.pdf")
