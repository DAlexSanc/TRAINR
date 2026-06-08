' launcher.vbs  —  TRAINR
' Launches main.py using pythonw.exe so no console window appears.
' Place this file at the install root alongside the Scripts/ folder and venv/.

Dim shell, dir
Set shell = CreateObject("WScript.Shell")

' Resolve the directory this .vbs lives in
dir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))

Dim pythonw, script
pythonw = dir & "venv\Scripts\pythonw.exe"
script  = dir & "Scripts\main.py"

' Check that the venv exists — show a helpful message if setup hasn't been run
Dim fso
Set fso = CreateObject("Scripting.FileSystemObject")
If Not fso.FileExists(pythonw) Then
    MsgBox "TRAINR environment not found." & vbCrLf & vbCrLf & _
           "Please run setup_env.ps1 first to set up the Python environment." & vbCrLf & vbCrLf & _
           "Expected location:" & vbCrLf & pythonw, _
           vbCritical, "TRAINR — Setup Required"
    WScript.Quit 1
End If

' Launch — WindowStyle 0 = hidden (no console flash)
shell.Run """" & pythonw & """ """ & script & """", 0, False

Set shell = Nothing
Set fso   = Nothing
