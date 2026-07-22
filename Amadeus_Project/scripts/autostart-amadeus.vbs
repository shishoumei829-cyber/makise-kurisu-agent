'use strict';
// Hidden launcher for Windows Startup — no console flash
var sh = new ActiveXObject('WScript.Shell');
var fso = new ActiveXObject('Scripting.FileSystemObject');
var scriptDir = fso.GetParentFolderName(WScript.ScriptFullName);
var ps1 = scriptDir + '\\autostart-amadeus.ps1';
var cmd = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + ps1 + '"';
sh.Run(cmd, 0, false);
