$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$XeLaTeX = "D:\texlive\2026\bin\windows\xelatex.exe"
$OutDir = Join-Path $ProjectRoot "output\pdf"
$TmpDir = Join-Path $ProjectRoot "tmp\pdfs\phd-application-handbook"
$TmpDirForTeX = $TmpDir.Replace("\", "/")

New-Item -ItemType Directory -Force -Path $OutDir, $TmpDir | Out-Null

& $Python (Join-Path $PSScriptRoot "build_handbook.py")
if ($LASTEXITCODE -ne 0) { throw "Content generation failed" }

Push-Location $PSScriptRoot
try {
  & $XeLaTeX "-output-directory=$TmpDirForTeX" -interaction=batchmode -halt-on-error -file-line-error research_handbook.tex
  if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass 1 failed" }
  & $XeLaTeX "-output-directory=$TmpDirForTeX" -interaction=batchmode -halt-on-error -file-line-error research_handbook.tex
  if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass 2 failed" }
} finally {
  Pop-Location
}

Copy-Item -LiteralPath (Join-Path $TmpDir "research_handbook.pdf") -Destination (Join-Path $OutDir "phd-application-research-handbook.pdf") -Force
Write-Output (Join-Path $OutDir "phd-application-research-handbook.pdf")
