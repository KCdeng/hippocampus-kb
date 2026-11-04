Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "kb_serve" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "killed " + $_.ProcessId }
