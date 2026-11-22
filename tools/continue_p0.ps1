$ErrorActionPreference = 'Stop'
Set-Location 'D:\IIITH A2\assignment_2b_emotion'
$env:PYTHONPATH = 'D:\IIITH A2\assignment_2b_emotion\src'
$env:HF_HUB_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
while (Get-Process -Id 2736 -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 15 }
$legacy = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'models[/\\](text|fusion)_pipeline[/\\](train|test).py' }
if ($legacy) { throw 'Legacy text/fusion queue is active. Do not launch a competing queue.' }
if (Test-Path 'Results/audits/process_collision/incident.json') { throw 'Resolve recorded ownership/provenance collision before launching this queue.' }
foreach ($entry in @(@('text','configs/text.yaml'), @('fusion','configs/fusion_concat.yaml'), @('fusion','configs/fusion.yaml'))) {
 foreach ($protocol in @('A','B','C')) {
  & 'D:\IIITH A2\.venv\Scripts\python.exe' -u -B -m mer.training.p0_train --mode $entry[0] --config $entry[1] --protocol $protocol
  if ($LASTEXITCODE -ne 0) { throw "Failed $entry protocol $protocol" }
 }
}
