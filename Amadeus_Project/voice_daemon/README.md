# Amadeus Voice Daemon

本地常开麦克风服务：VAD 切句、声纹验证、ASR 转写，然后把可信片段发给后端 `/ambient-hearing`。

## 安装

```powershell
cd D:\Amadeus_Trae\Amadeus_Project
py -3.12 -m venv .venv-voice
.\.venv-voice\Scripts\pip install -U pip
.\.venv-voice\Scripts\pip install -r voice_daemon\requirements.txt
```

## 注册你的声纹

在安静环境读 20-30 秒中文，越自然越好：

```powershell
.\.venv-voice\Scripts\python voice_daemon\amadeus_voice_daemon.py enroll --seconds 25 --speaker owner
```

## 启动常开耳朵

```powershell
.\.venv-voice\Scripts\python voice_daemon\amadeus_voice_daemon.py listen --speaker owner
```

可选参数：

```powershell
.\.venv-voice\Scripts\python voice_daemon\amadeus_voice_daemon.py devices
.\.venv-voice\Scripts\python voice_daemon\amadeus_voice_daemon.py listen --device 1
```

默认只把 `verified=true` 的 owner 语音发送给后端；陌生人会被丢弃。需要调试时加 `--post-unverified`。
