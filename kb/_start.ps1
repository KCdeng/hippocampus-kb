# 启动 8310 常驻检索服务。两条硬规矩都是踩过坑加的：
#   1) 端口已有监听者就**立刻退出 1，一个日志文件都不碰**。Start-Process 的
#      -RedirectStandardOutput/Error 用 CREATE_ALWAYS 打开文件，进程创建那一刻就把上一世的
#      现场截成 0 字节——实测：端口被占、新实例自己 FATAL 退出时同样截（在跑服务的 120 字节
#      启动日志被死实例的一行 FATAL 顶掉）。所以防截断必须做在碰文件之前。
#   2) 真要点火前，把 serve.out.log / serve.err.log 带时间戳归档（留 3 份），这样重启后
#      还能查上一次崩溃前的 BOOT FAILED / RELOAD FAILED traceback。归档失败就拒绝启动：
#      宁可不动，也不能把唯一的现场悄悄清掉。
$ErrorActionPreference = 'Stop'
# 控制台默认按 GBK 编码写出，被重定向时中文提示会变成乱码——"refused/归档失败"这种
# 关键提示读不了，等于没有。脚本自身是 UTF-8 BOM，输出也钉成 UTF-8。
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$here = $PSScriptRoot
$py = Join-Path (Split-Path $here) 'venv\Scripts\python.exe'
$port = if ($env:KB_PORT) { [int]$env:KB_PORT } else { 8310 }

function Test-PortBusy($p) {
  $c = New-Object System.Net.Sockets.TcpClient
  try {
    $ar = $c.BeginConnect('127.0.0.1', $p, $null, $null)
    return ($ar.AsyncWaitHandle.WaitOne(800) -and $c.Connected)
  } catch {
    return $false
  } finally {
    $c.Close()
  }
}

if (Test-PortBusy $port) {
  Write-Output ('refused: 127.0.0.1:' + $port + ' 已有服务在监听，本脚本未改动任何日志文件')
  exit 1
}

foreach ($log in @('serve.out.log', 'serve.err.log')) {
  $path = Join-Path $here $log
  if (-not (Test-Path -LiteralPath $path)) { continue }
  & $py -X utf8 (Join-Path $here 'kb_log_archive.py') archive $path --keep 3
  if ($LASTEXITCODE -ne 0) {
    Write-Output ('refused: 归档 ' + $log + ' 失败（退出码 ' + $LASTEXITCODE + '），拒绝以截断旧日志的方式启动')
    exit 1
  }
}

Start-Process -FilePath $py -ArgumentList '-X', 'utf8', 'kb_serve.py' `
  -WorkingDirectory $here -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $here 'serve.out.log') `
  -RedirectStandardError (Join-Path $here 'serve.err.log')
Write-Output ('started ' + $py)
exit 0
