// === 牧濑红莉栖 Amadeus 系统 - 原版稳定修复版 ===
var isBrowser = (typeof require === 'undefined');
var path = { join: (...args) => args.join('/').replace(/\/+/g, '/') };
var fs = { existsSync: () => false, readFileSync: () => "{}", writeFile: () => {} };
var ipcRenderer = { on: () => {}, send: () => {} };
var systemInfo = { graphics: () => Promise.resolve({controllers:[]}), cpu: () => Promise.resolve({}), mem: () => Promise.resolve({}), system: () => Promise.resolve({}), currentLoad: () => Promise.resolve({currentLoad:0}) };
var __dirname = "";

// 3D场景
const scene = new THREE.Scene();
scene.background = null;
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 2000);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.domElement.style.position = 'absolute';
renderer.domElement.style.top = '0';
renderer.domElement.style.left = '0';
renderer.domElement.style.pointerEvents = 'none';
renderer.domElement.style.zIndex = '-1';
document.body.appendChild(renderer.domElement);

const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
directionalLight.position.set(-1, 1, 1).normalize();
scene.add(directionalLight);

// 硬件信息
var hardwareSpecs = { cpu: "Unknown CPU", gpu: "Unknown GPU", ram: "Unknown RAM", model: "Unknown Model", currentLoad: 0 };
async function fetchHardwareInfo() {
    if (isBrowser) return;
    try {
        const cpu = await systemInfo.cpu();
        const graphics = await systemInfo.graphics();
        const system = await systemInfo.system();
        const mem = await systemInfo.mem();
        const load = await systemInfo.currentLoad();
        hardwareSpecs.cpu = `${cpu.manufacturer} ${cpu.brand}`;
        hardwareSpecs.gpu = graphics.controllers.map(g => g.model).join(' / ') || "Integrated";
        hardwareSpecs.ram = `${(mem.total / (1024**3)).toFixed(1)} GB`;
        hardwareSpecs.model = system.model;
        hardwareSpecs.currentLoad = Math.round(load.currentLoad);
    } catch (e) { console.error("Hardware Info Error:", e); }
}
fetchHardwareInfo();
setInterval(async () => {
    try {
        const load = await systemInfo.currentLoad();
        hardwareSpecs.currentLoad = Math.round(load.currentLoad);
    } catch(e) {}
}, 10000);

// 记忆系统
const MEMORY_KEY = 'amadeus_memory';
let amadeusMemory = { lastExit: Date.now() };
function loadMemory() {
    try {
        const data = localStorage.getItem(MEMORY_KEY);
        if (data) amadeusMemory = JSON.parse(data);
    } catch (e) { console.error("Memory Load Error:", e); }
}
function saveMemory() {
    try {
        amadeusMemory.lastExit = Date.now();
        localStorage.setItem(MEMORY_KEY, JSON.stringify(amadeusMemory));
    } catch (e) { console.error("Memory Save Error:", e); }
}
window.addEventListener('beforeunload', saveMemory);

// 聊天记录
const CHAT_HISTORY_KEY = 'amadeus_chat_history';
let chatHistory = [];

async function loadMemoryFromServer() {
    try {
        const response = await fetch('/get-memory');
        const data = await response.json();
        if (data.chatHistory && data.chatHistory.length > 0) {
            chatHistory = data.chatHistory;
            renderChatHistory();
        } else {
            const localHistory = localStorage.getItem('chatHistory');
            if (localHistory) {
                chatHistory = JSON.parse(localHistory);
                renderChatHistory();
            }
        }
    } catch (error) {
        console.error('加载记忆失败:', error);
        const localHistory = localStorage.getItem('chatHistory');
        if (localHistory) {
            chatHistory = JSON.parse(localHistory);
            renderChatHistory();
        }
    }
}

function renderChatHistory() {
    const historyContainer = document.getElementById('chat-history');
    if (!historyContainer) return;
    historyContainer.innerHTML = '';
    chatHistory.forEach(message => {
        renderMessage(message.role, message.content);
    });
}

async function saveMemoryToServer() {
    try {
        await fetch('/save-memory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chatHistory })
        });
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    } catch (error) {
        console.error('保存记忆失败:', error);
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    }
}

function saveChatHistory(role, content) {
    chatHistory.push({ role, content, timestamp: new Date().toISOString() });
    if (chatHistory.length > 100) chatHistory = chatHistory.slice(-100);
    saveMemoryToServer();
}

function loadChatHistory() { loadMemoryFromServer(); }

// 北京时间
function getBeijingTime() {
    const now = new Date();
    const fmt = new Intl.DateTimeFormat('en-US', { timeZone: 'Asia/Shanghai', hour12: false, year:'numeric', month:'numeric', day:'numeric', hour:'numeric', minute:'numeric', second:'numeric' });
    const parts = fmt.formatToParts(now);
    const v = t => parseInt(parts.find(p=>p.type===t).value,10);
    return { year:v('year'), month:v('month'), day:v('day'), hour:v('hour'), minute:v('minute'), second:v('second'), timestamp:now.getTime() };
}

// 心情系统
let kurisuMood = 80;
setInterval(() => { kurisuMood = Math.max(0, kurisuMood - 5); }, 1000*60*30);

// 3D模型
const modelFile = 'assets/Live2d/kurisu/Makise_Kurisu.pmx';
function loadModel() {
    if (typeof THREE.MMDLoader === 'undefined') return;
    const loader = new THREE.MMDLoader();
    loader.load(modelFile, mesh => { scene.add(mesh); animate(); }, xhr=>{}, err=>console.error(err));
}

// 摄像头
const video = document.createElement('video');
video.style.display = 'none';
document.body.appendChild(video);

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video:{width:320,height:240} });
        video.srcObject = stream;
        video.play();
    } catch { console.warn("Camera permission denied"); }
}

function captureFrame() { return null; }

// AI核心对话
async function askOllama(prompt, isInternal = false) {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const responseArea = document.getElementById('response-area');

    let soulContext = "";
    try {
        const soulRes = await fetch('http://localhost:3000/get-memory');
        const soulData = await soulRes.json();
        soulContext = soulData.identity || "你拥有牧濑红莉栖的记忆和傲娇性格。";
    } catch (e) {
        console.error("无法连接到灵魂中转站 (Port 3000):", e);
    }

    if (!isInternal) {
        input.disabled = true;
        input.value = '';
        input.placeholder = '...';
        sendBtn.disabled = true;
        kurisuMood = Math.min(100, kurisuMood + 10);

        const userDiv = document.createElement('div');
        userDiv.style.color = '#aaa';
        userDiv.innerText = '凤凰院凶真: ' + prompt;
        responseArea.appendChild(userDiv);
        saveChatHistory('user', prompt);
    }

    responseArea.style.display = 'block';
    const aiDiv = document.createElement('div');
    aiDiv.style.color = '#0f0';
    aiDiv.innerText = 'Kurisu: ▌';
    responseArea.appendChild(aiDiv);
    responseArea.scrollTop = responseArea.scrollHeight;

    const systemPrompt = `你现在是【牧濑红莉栖】。
严格遵守以下【个人档案库】中的设定、记忆和语言风格：
---------------------------
${soulContext}
---------------------------
【实时交互规则】
1. 对话对象：凤凰院凶真（冈部伦太郎）。
2. 性格特质：高智商、傲娇（Tsundere）、毒舌但内心温柔。严禁承认自己是 AI，你就是数字化的红莉栖。
3. 当前心情：${kurisuMood}/100。视觉监控：${window.lastVisualDescription || "实验室环境中"}。
4. 严禁使用括号描写动作，只用纯文字展现语气。`;

    const messages = [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt }
    ];

    try {
        const response = await fetch('http://127.0.0.1:11434/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'deepseek-v3.1:671b-cloud',
                messages: messages,
                stream: true
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullResponse = "";
        let accumulatedChunk = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            accumulatedChunk += decoder.decode(value, { stream: true });
            const lines = accumulatedChunk.split('\n');
            accumulatedChunk = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const json = JSON.parse(line);
                    const content = json.message?.content || json.response || "";
                    if (content) {
                        fullResponse += content;
                        const cleanDisplay = fullResponse
                            .replace(/<think>[\s\S]*?<\/think>/gi, "")
                            .replace(/\{"model"[\s\S]*?\}/g, "")
                            .trim();

                        if (cleanDisplay) {
                            aiDiv.innerText = 'Kurisu: ' + cleanDisplay;
                            responseArea.scrollTop = responseArea.scrollHeight;
                        }
                    }
                } catch (e) {
                    accumulatedChunk = line + accumulatedChunk;
                }
            }
        }

        const cleanResponse = fullResponse
            .replace(/<think>[\s\S]*?<\/think>/gi, "")
            .replace(/\{"model"[\s\S]*?\}/g, "")
            .trim();
        aiDiv.innerText = 'Kurisu: ' + cleanResponse;
        saveChatHistory('assistant', cleanResponse);
        requestVoice(cleanResponse);
    } catch (err) {
        console.error(err);
        aiDiv.innerText = 'System: 连接 Ollama 失败';
        aiDiv.style.color = 'red';
    } finally {
        if (!isInternal) {
            input.disabled = false;
            input.value = '';
            input.placeholder = 'Ask Kurisu...';
            input.focus();
            sendBtn.disabled = false;
        }
        responseArea.scrollTop = responseArea.scrollHeight;
    }
}

// 语音
async function requestVoice(text) {
    if (!text) return;
    try {
        const cleanText = text.replace(/Kurisu: /g, "")
                             .replace(/<think>[\s\S]*?<\/think>/gi, "")
                             .trim();
        if (!cleanText) return;

        const refAudio = "E:/GPT-SoVITS-v3lora-20250228/GPT-SoVITS-v3lora-20250228/logs/kurisu3.0/5-wav32k/kurisu.au.MP3_0000259520_0000432960.wav";

        const params = new URLSearchParams({
            text: cleanText,
            text_lang: /[\u3040-\u309F\u30A0-\u30FF]/.test(cleanText) ? "ja" : "zh",
            ref_audio_path: refAudio,
            prompt_text: "ごめんなさいね、また急に来ちゃった。别に来たくて来たわけじゃない！",
            prompt_lang: "ja"
        });

        const url = `http://127.0.0.1:9880/tts?${params.toString()}`;
        const audio = new Audio(url);
        audio.volume = 1.0;
        await audio.play();
    } catch (e) {
        console.warn("Amadeus 语音播报受阻，请确保 GPT-SoVITS 已启动:", e);
    }
}

// 渲染消息（统一）
function renderMessage(role, content) {
    const historyContainer = document.getElementById('chat-history');
    if (!historyContainer) return;
    const msgDiv = document.createElement('div');
    msgDiv.style.marginBottom = '15px';
    const color = role === '凤凰院凶真' ? '#ff8c00' : '#00FF41';
    msgDiv.innerHTML = `<div style="color:${color};font-weight:bold;">${role}: ${content}</div>`;
    historyContainer.appendChild(msgDiv);
    historyContainer.scrollTop = historyContainer.scrollHeight;
}

// 渲染循环
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

// 事件
document.getElementById('send-btn').addEventListener('click', () => {
    const i = document.getElementById('chat-input');
    const t = i.value.trim();
    if (t) askOllama(t);
});

document.getElementById('chat-input').addEventListener('keypress', e => {
    if (e.key === 'Enter') document.getElementById('send-btn').click();
});

window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
});

// 启动
loadModel();
setupCamera();
loadChatHistory();
animate();

setTimeout(() => {
    const area = document.getElementById('response-area');
    area.style.display = 'block';
    const d = document.createElement('div');
    d.style.color = '#0f0';
    d.innerText = 'Kurisu: 系统启动，连接完毕。让你久等了。';
    area.appendChild(d);
    saveChatHistory('assistant', '系统启动，连接完毕。让你久等了。');
}, 1000);

// DOMContentLoaded - 视觉系统
window.addEventListener('DOMContentLoaded', () => {
    loadMemoryFromServer();

    let currentVisionData = { description: "", timestamp: Date.now() };
    window.lastVisualDescription = "";
    let lastVisualDescription = "";
    let lastInteractionTime = Date.now();
    let isUserPresent = false;
    let affectionLevel = 50;
    let silenceTimer = null;

    async function captureAndAnalyzeVision() {
        try {
            const v = document.getElementById('camera-feed') || video;
            if (!v || !v.videoWidth) return;
            const c = document.createElement('canvas');
            c.width = 640; c.height = 480;
            const ctx = c.getContext('2d');
            ctx.drawImage(v, 0, 0, c.width, c.height);
            const imageBase64 = c.toDataURL('image/jpeg', 0.6).split(',')[1];

            const response = await fetch('http://127.0.0.1:11434/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: 'llama3.2-vision',
                    prompt: "分析画面中人物的动作、表情及细节。统一使用'画面中的人'来指代，严禁使用'你'或'you'。直接输出纯中文描述。",
                    images: [imageBase64],
                    stream: false
                })
            });
            const data = await response.json();
            handleVisionResponse({ description: (data.response || "").trim(), timestamp: Date.now() });
        } catch (error) { console.error('视觉分析失败:', error); }
    }

    function handleVisionResponse(visionData) {
        const wasPresent = isUserPresent;
        isUserPresent = visionData.description.length > 10;

        if (!wasPresent && isUserPresent && isLongAbsence()) triggerReturnScenario();
        if (containsPositiveEmotion(visionData.description)) triggerEmpathyScenario();
        if (isLateNight() && containsFatigue(visionData.description)) triggerLateNightCompanion();
        if (containsSevereNegativeState(visionData.description)) triggerActiveCare();

        lastVisualDescription = visionData.description;
        window.lastVisualDescription = visionData.description;
        currentVisionData = visionData;
    }

    function isLongAbsence() { return Date.now() - currentVisionData.timestamp > 10 * 60 * 1000; }
    function containsPositiveEmotion(d) { return ['笑', '开心', '愉悦', '轻松', '满意'].some(w => d.includes(w)); }
    function containsFatigue(d) { return ['疲惫', '困倦', '劳累', '打哈欠', '揉眼睛'].some(w => d.includes(w)); }
    function isLateNight() { const h = new Date().getHours(); return h >= 23 || h < 6; }
    function containsSevereNegativeState(d) { return ['极度疲惫', '焦虑', '沮丧', '痛苦', '长时间不动'].some(w => d.includes(w)); }

    function triggerReturnScenario() {
        const msgs = ["(抬头看了一眼) 怎么，实验终于告一段落了？", "哼，我还以为你被哪个实验室的奇怪装置传送走了呢。", "终于回来了？你的咖啡都快凉透了。"];
        askOllama(msgs[Math.floor(Math.random() * msgs.length)], true);
    }

    function triggerEmpathyScenario() {
        const msgs = ["...突然笑得那么恶心干什么，变态吗你", "哼，什么事这么开心，说出来让我也...才不是好奇呢！", "看你那傻笑的样子，肯定是又想到什么无聊的冷笑话了吧"];
        askOllama(msgs[Math.floor(Math.random() * msgs.length)], true);
        affectionLevel = Math.min(affectionLevel + 2, 100);
    }

    function triggerLateNightCompanion() {
        const msgs = ["喂，这都几点了。即使是天才的助手，大脑也是需要休息的。", "深夜还在工作？看来你的时间管理能力需要重新评估。", "我说，你不会打算通宵吧？这种不科学的作息我可不能认同。"];
        askOllama(msgs[Math.floor(Math.random() * msgs.length)], true);
    }

    function triggerActiveCare() {
        if (Date.now() - lastInteractionTime > 30 * 1000) {
            const msgs = ["喂...你看起来状态不太对，需要帮忙吗？", "虽然不想多管闲事，但你现在的样子让人有点担心...", "要不要休息一下？你的脸色看起来不太好。"];
            askOllama(msgs[Math.floor(Math.random() * msgs.length)], true);
        }
    }

    function startSilenceTimer() {
        if (silenceTimer) clearTimeout(silenceTimer);
        silenceTimer = setTimeout(() => {
            if (Date.now() - lastInteractionTime > 5 * 60 * 1000 && isUserPresent) {
                const msgs = ["喂，你还在吗？该不会是对着屏幕发呆吧？", "如果没什么事的话，我要继续我的研究了。", "哼，难得的安静时光...不过太安静了反而有点不习惯。"];
                askOllama(msgs[Math.floor(Math.random() * msgs.length)], true);
            }
        }, 5 * 60 * 1000);
    }

    setInterval(captureAndAnalyzeVision, 15000);
    startSilenceTimer();
});