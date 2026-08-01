' Launch Amadeus Pure Window on Windows Startup
Dim sh, fso, scriptDir, bat, cmd
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
bat = scriptDir & "\launch-pure-window.bat"
If Not fso.FileExists(bat) Then
  WScript.Quit 1
End If
cmd = "cmd.exe /c """ & bat & """"
' 0 = hide console flash from the bat
sh.Run cmd, 0, False
