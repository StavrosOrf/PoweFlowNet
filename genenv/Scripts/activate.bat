@echo off

rem This file is UTF-8 encoded, so we need to update the current code page while executing it
for /f "tokens=2 delims=:." %%a in ('"%SystemRoot%\System32\chcp.com"') do (
    set _OLD_CODEPAGE=%%a
)
if defined _OLD_CODEPAGE (
    "%SystemRoot%\System32\chcp.com" 65001 > nul
)

<<<<<<< Updated upstream:genenv/Scripts/activate.bat
set VIRTUAL_ENV=C:\Users\siddh\OneDrive\Desktop\Sid\Work\Q4\ML4GD\Project\PoweFlowNet\genenv
=======
<<<<<<< Updated upstream
<<<<<<< Updated upstream
set VIRTUAL_ENV=C:\Users\siddh\OneDrive\Desktop\Sid\Work\Q4\ML4GD\Project\PoweFlowNet\PowPyEnv
=======
set VIRTUAL_ENV=C:\Users\siddh\OneDrive\Desktop\Sid\Work\Q4\ML4GD\Project\PoweFlowNet\Powpyenv
>>>>>>> Stashed changes
=======
set VIRTUAL_ENV=C:\Users\siddh\OneDrive\Desktop\Sid\Work\Q4\ML4GD\Project\PoweFlowNet\Powpyenv
>>>>>>> Stashed changes
>>>>>>> Stashed changes:PowPyEnv/Scripts/activate.bat

if not defined PROMPT set PROMPT=$P$G

if defined _OLD_VIRTUAL_PROMPT set PROMPT=%_OLD_VIRTUAL_PROMPT%
if defined _OLD_VIRTUAL_PYTHONHOME set PYTHONHOME=%_OLD_VIRTUAL_PYTHONHOME%

set _OLD_VIRTUAL_PROMPT=%PROMPT%
<<<<<<< Updated upstream:genenv/Scripts/activate.bat
set PROMPT=(genenv) %PROMPT%
=======
<<<<<<< Updated upstream
<<<<<<< Updated upstream
set PROMPT=(PowPyEnv) %PROMPT%
=======
set PROMPT=(Powpyenv) %PROMPT%
>>>>>>> Stashed changes
=======
set PROMPT=(Powpyenv) %PROMPT%
>>>>>>> Stashed changes
>>>>>>> Stashed changes:PowPyEnv/Scripts/activate.bat

if defined PYTHONHOME set _OLD_VIRTUAL_PYTHONHOME=%PYTHONHOME%
set PYTHONHOME=

if defined _OLD_VIRTUAL_PATH set PATH=%_OLD_VIRTUAL_PATH%
if not defined _OLD_VIRTUAL_PATH set _OLD_VIRTUAL_PATH=%PATH%

set PATH=%VIRTUAL_ENV%\Scripts;%PATH%
<<<<<<< Updated upstream:genenv/Scripts/activate.bat
set VIRTUAL_ENV_PROMPT=(genenv) 
=======
<<<<<<< Updated upstream
<<<<<<< Updated upstream
set VIRTUAL_ENV_PROMPT=(PowPyEnv) 
=======
set VIRTUAL_ENV_PROMPT=(Powpyenv) 
>>>>>>> Stashed changes
=======
set VIRTUAL_ENV_PROMPT=(Powpyenv) 
>>>>>>> Stashed changes
>>>>>>> Stashed changes:PowPyEnv/Scripts/activate.bat

:END
if defined _OLD_CODEPAGE (
    "%SystemRoot%\System32\chcp.com" %_OLD_CODEPAGE% > nul
    set _OLD_CODEPAGE=
)
