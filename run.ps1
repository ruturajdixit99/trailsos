# Windows helper (no make needed):  .\run.ps1 [run|test|bench|install]
param([string]$task = "run")
$py = ".\.venv\Scripts\python.exe"
switch ($task) {
  "install" { python -m venv .venv; & $py -m pip install -r requirements.txt }
  "test"    { & $py -m pytest -q -s -p no:warnings }
  "bench"   { & $py scripts\benchmark.py }
  default   { Start-Process "http://127.0.0.1:8000"; & $py -m uvicorn server.app:app --port 8000 }
}
