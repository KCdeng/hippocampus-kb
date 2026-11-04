$t = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "kb_serve" }
foreach ($x in $t) { Stop-Process -Id $x.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Milliseconds 800
if (@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "kb_serve" }).Count -eq 0) { Write-Output "all-stopped" } else { Write-Output "still-alive" }
