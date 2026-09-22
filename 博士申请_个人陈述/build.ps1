$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$BuildDir = Join-Path $ProjectDir "build"
$XeLaTeX = "D:\texlive\2026\bin\windows\xelatex.exe"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$BuildDirForTeX = $BuildDir.Replace("\", "/")
Push-Location $ProjectDir
try {
  & $XeLaTeX "-output-directory=$BuildDirForTeX" -interaction=batchmode -halt-on-error -file-line-error main.tex
  if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass 1 failed" }
  & $XeLaTeX "-output-directory=$BuildDirForTeX" -interaction=batchmode -halt-on-error -file-line-error main.tex
  if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass 2 failed" }
} finally {
  Pop-Location
}
Copy-Item -LiteralPath (Join-Path $BuildDir "main.pdf") -Destination (Join-Path $ProjectDir "个人陈述-马睿骁.pdf") -Force
Write-Output (Join-Path $ProjectDir "个人陈述-马睿骁.pdf")