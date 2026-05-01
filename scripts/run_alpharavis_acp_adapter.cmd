@echo off
setlocal

if "%ALPHARAVIS_ACP_REPO%"=="" (
  set "ALPHARAVIS_ACP_REPO=%~dp0.."
)

python "%ALPHARAVIS_ACP_REPO%\langgraph-app\alpharavis_acp_adapter.py"
