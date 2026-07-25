'use strict';

const { inferUserBdi } = require('../bdi_engine');
const { inferMainEventFromInput, updatePAD, savePAD } = require('../cognitive/pad');
const { getReplyLanguageMode } = require('../lib/replyLanguage');
const {
  _clipInnerPrompt, symbolicReasoning, buildPrompt,
} = require('../cognitive/prompts');
const {
  utteranceFocusLine,
  isContinuityFollowup,
  resolveFollowupForModel,
  buildContinuityRepairMessages,
  filterRagHits,
  filterAutonomyRagHits,
  filterAutonomyMemCtx,
  buildEngagementHint,
  stripOrphanClosingSentence,
  stripChatMarkdown,
  stripRoleplayActions,
  shouldReplaceStreamText,
} = require('../cognitive/replyAlign');
const {
  buildOllamaMessages,
  fitSystemForDialogue,
  estimateMessageChars,
  resolvePromptCharBudget,
  isFastConversationTurn,
} = require('../cognitive/chatTurns');
const { derivePresence, presenceToPromptLine } = require('../cognitive/presence');
const { buildTurnStyleBlock } = require('../cognitive/turnStyle');
const {
  buildCompanionBlock,
  effectiveRelScore,
  idleSilencePadDelta,
} = require('../cognitive/companionMode');
const {
  ensureWhoamiOnDisk,
  buildPartnerContextBlock,
  partnerIsOkabe,
  resolvePartnerDisplayName,
} = require('../lib/partnerIdentity');
const {
  buildProactiveReplyFocus,
  buildAutonomyContinuityBlock,
} = require('../cognitive/turnContinuity');
const { normalizeClientContext, buildClientContextBlock } = require('../lib/clientContext');
const { buildSocialIdentityPrompt } = require('../cognitive/socialIdentity');
const { buildExpressionVariantBlock } = require('../cognitive/expressionVariants');
const { EmotionalBandwidthEngine } = require('../cognitive/emotionalBandwidth');
const { needsConversationRecall } = require('../lib/unifiedDialogueLog');
const userPresence = require('../lib/userPresence');
const { perceiveIncomingChat } = require('./perceive');
const { ChatTurnRegistry } = require('../lib/chatTurnRegistry');
const { stripAgencyTaskMarkers } = require('../lib/butler/intent');

async function _acceptAgencyTaskFromReply(d, rawText, { turnId } = {}) {
  if (!d?.butlerKernel?.consumeAgencyReply) {
    return { spoken: stripAgencyTaskMarkers(rawText), task: null, created: false };
  }
  const accepted = d.butlerKernel.consumeAgencyReply(rawText, {
    source: 'agency',
    turnId,
  });
  const task = accepted.task;
  if (task && accepted.created) {
    try {
      const planned = await d.butlerKernel.planTask(task.id, { allowStrong: false });
      if (planned?.task?.status === 'ready' && planned.plan?.source === 'heuristic') {
        await d.butlerKernel.runPlan(task.id, { source: 'agency' });
      } else if (!planned && ['proposed', 'waiting_confirmation'].includes(
        d.butlerKernel.tasks.getTask(task.id)?.status || task.status,
      )) {
        d.butlerKernel.planTask(task.id, { allowStrong: true, source: 'background' }).catch((error) => {
          d.butlerKernel.failPlanning?.(task.id, error, { followup: true });
          d.agentDebugLog?.({
            hypothesisId: 'butler-agency-background-plan',
            location: 'brain/legacyChat',
            message: error.message,
            data: { taskId: task.id },
          });
        });
      }
    } catch (error) {
      d.agentDebugLog?.({
        hypothesisId: 'butler-agency-plan',
        location: 'brain/legacyChat',
        message: error.message,
        data: { taskId: task.id },
      });
    }
  }
  return accepted;
}
let _deps = null;
const chatTurnRegistry = new ChatTurnRegistry();

function init(deps) {
  _deps = deps;
}

/**
 * 完整 /chat 主链（自 server.js 迁入，阶段 1 legacy）。
 */
async function runChatTurn(req, res) {
  const d = _deps;
  if (!d) throw new Error('[brain/legacyChat] init(deps) required');
  const s = d.state;
  const lease = chatTurnRegistry.begin(req.body?.conversationId, req.body?.turnId);
  const cancelOnDisconnect = () => {
    if (!res.writableEnded) lease.cancel('client-disconnected');
  };
  res.once('close', cancelOnDisconnect);

  const __streamRequested = req.body && req.body.stream === true;
  try {
    let model      = req.body.model || process.env.AMADEUS_CHAT_MODEL || 'kurisu:latest';
    const useStream= req.body.stream === true;  // 默认非流式
    let temp       = req.body.temperature ?? (Number(process.env.AMADEUS_CHAT_TEMP) || 0.72);
    let maxTok   = req.body.max_tokens  ?? 384;

    const {
      parsed,
      userContent: perceivedUserContent,
      autonomyInitiative,
      idleMsSinceUser,
      replyingToProactive,
      proactiveAnchor,
      recentUserLinesForMem,
      lastRealUserLine,
      userPresenceState,
      useLongTermMemory,
      useRagForTurn,
      cognitiveInput,
    } = perceiveIncomingChat(req.body, d);
    let userContent = perceivedUserContent;
    const fastConversation = isFastConversationTurn(userContent, {
      autonomy: autonomyInitiative,
      useLongTermMemory,
      hasTask: d.butlerKernel?.getActiveTasks?.().some((task) => task.description === userContent),
    });
    // kurisu 常先烧 thinking；96 会整轮空 content。快对话只压缩上下文，不砍生成长度。
    if (fastConversation) maxTok = Math.min(Math.max(Number(maxTok) || 0, 320), 512);
    let autonomyDecision = null;
    let dialogueForOllama = [];

    const histCap = Number(process.env.AMADEUS_CHAT_HISTORY_MSGS);
    const maxHist = Number.isFinite(histCap) && histCap > 0 ? histCap : 12;

    const replyLanguage = getReplyLanguageMode();

    // 单一 prompt 源：忽略前端 system，仅用服务端 soul
    let systemContent = d.cachedSoulContent;
    if (replyLanguage === 'ja') {
      systemContent = [d.cachedSoulContent, d.cachedVoiceContent, d.cachedCharacterRules].filter(Boolean).join('\n\n');
    } else if (d.cachedCharacterRules) {
      systemContent = systemContent + '\n\n' + d.cachedCharacterRules;
    }
    systemContent = systemContent.trim();

    // 用户原句只观察偏好/确认，不按词表进队。进行中的任务事实可注入；动手登记等她回复里的标记。
    if (!autonomyInitiative && d.butlerKernel) {
      const butlerObservation = d.butlerKernel.observeUserRequest({
        text: userContent,
        source: 'chat',
        turnId: lease.turnId,
      });
      if (butlerObservation.action === 'confirmation_approved' && butlerObservation.task?.status === 'ready') {
        try {
          const execution = await d.butlerKernel.runPlan(butlerObservation.task.id, { source: 'chat' });
          const taskBlock = d.butlerKernel.taskPromptBlock(butlerObservation.task.id, execution);
          if (taskBlock) systemContent = `${systemContent}\n\n${taskBlock}`;
        } catch (error) {
          d.agentDebugLog?.({
            hypothesisId: 'butler-confirm-run',
            location: 'brain/legacyChat',
            message: error.message,
            data: { taskId: butlerObservation.task.id },
          });
        }
      } else if (butlerObservation.promptBlock) {
        systemContent = `${systemContent}\n\n${butlerObservation.promptBlock}`;
      } else {
        const active = d.butlerKernel.getActiveTasks?.() || [];
        if (active[0]) {
          const taskBlock = d.butlerKernel.taskPromptBlock(active[0].id);
          if (taskBlock) systemContent = `${systemContent}\n\n${taskBlock}`;
        }
      }
      systemContent = `${systemContent}\n\n${d.butlerAgencyHint || [
        '【动手登记】只有当你决定亲自去办一件可用工具完成的现实事（开应用、找文件、设提醒、查系统等）时，在台词末尾另起一行写：',
        '⟦AMADEUS_TASK{"title":"短标题","description":"要办的事"}⟧',
        '社交请求、给他派任务玩、闲聊、安慰、抬杠：不要写这一行。没有这一行就不会启动执行。台词里不要念出标记本身。',
      ].join('\n')}`;
    }

    if (!userContent) {
      if (useStream) {
        res.setHeader('Content-Type','text/event-stream');
        res.write('data: [DONE]\n\n');
        return res.end();
      }
      return res.json({ response:'', choices:[{message:{role:'assistant',content:''}}] });
    }

    const synced = req.body.allowHistoryImport === true
      ? d.unifiedDialogueLog.syncFromDialogue(parsed.dialogue)
      : 0;
    if (synced > 0) {
      console.log(`[conversation] 从多轮历史补录 ${synced} 条`);
    }
    const userLine = cognitiveInput || userContent;
    const isRealUserTurn = !autonomyInitiative;
    const memoryAdmission = d.memoryAdmission.assessUserText(userLine, {
      source: isRealUserTurn ? 'user' : 'synthetic',
      synthetic: !isRealUserTurn,
    });
    if (isRealUserTurn) {
      d.memoryAdmission.observe('user', userLine);
      if (replyingToProactive) {
        d.conversationInitiative?.registerFeedback({ type: 'reply', text: userLine });
      }
    }
    const lastEntry = d.unifiedDialogueLog.getRecent(1)[0];
    const userNorm = String(userLine || '').trim();
    const alreadyLogged = lastEntry
      && lastEntry.role === 'user'
      && String(lastEntry.text || '').trim() === userNorm;
    if (isRealUserTurn && !alreadyLogged && userNorm) {
      d.unifiedDialogueLog.append('user', userLine, {
        conversationId: lease.conversationId,
        turnId: lease.turnId,
        source: 'chat',
      });
    }

    dialogueForOllama = d.unifiedDialogueLog.toOllamaDialogue({ maxMsgs: maxHist });
    if (userNorm && (!dialogueForOllama.length || dialogueForOllama[dialogueForOllama.length - 1].role !== 'user'
      || dialogueForOllama[dialogueForOllama.length - 1].content !== userNorm)) {
      dialogueForOllama.push({ role: 'user', content: userNorm });
      if (dialogueForOllama.length > maxHist) dialogueForOllama = dialogueForOllama.slice(-maxHist);
    }

    if (isRealUserTurn) s.chatTurnCounter++;

    if (autonomyInitiative && idleMsSinceUser > 0 && !userPresence.isPresenceActive(userPresenceState)) {
      const idlePad = idleSilencePadDelta(idleMsSinceUser, effectiveRelScore(d.memorySystem.getRelationshipScore()));
      if (idlePad) {
        s.currentPAD = updatePAD(s.currentPAD, idlePad, 0.35);
        savePAD(d.padPath, s.currentPAD);
        console.log(`[autonomy] 久未回复 PAD 波动 idleMin≈${(idleMsSinceUser / 60000).toFixed(1)} A+${idlePad.A.toFixed(2)}`);
      }
      const idleCycle = d.digitalLife.runIdleCycle({
        idleMs: idleMsSinceUser,
        pad: s.currentPAD,
        memorySystem: d.memorySystem,
        memory: d.memorySystem,
        motivationState: d.motivationState,
      });
      if (idleCycle.consolidated && idleCycle.insights.length) {
        console.log(`[memory-reorg] 整理洞察: ${idleCycle.insights.slice(0, 2).join('；')}`);
      }
      if (idleCycle.dream) {
        console.log(`[dream] ${idleCycle.dream.text.slice(0, 80)}`);
      }
      if (idleCycle.autonomy) {
        autonomyDecision = idleCycle.autonomy;
        console.log(`[autonomy-loop] ${autonomyDecision.action} speak=${autonomyDecision.shouldAct} urge=${autonomyDecision.primaryUrge?.drive || 'none'}`);
      }
    }

    // ══ ① 记忆系统：单一主事件 + PAD（不在此处落盘 pad_state）══
    const mainEvent = isRealUserTurn
      ? inferMainEventFromInput(cognitiveInput, s.currentPAD)
      : { type: 'neutral', content: '', importance: 0, delta: {} };
    const evDelta = mainEvent.delta || {};
    s.currentPAD = updatePAD(s.currentPAD, evDelta, mainEvent.importance);
    if (memoryAdmission.allowEvent && (mainEvent.type !== 'neutral' || mainEvent.importance > 0.2)) {
      d.memorySystem.addEvent(mainEvent.type, mainEvent.content, mainEvent.importance, evDelta);
    }
    d.updateMotivationFromMemory();

    if (isRealUserTurn) s.lastDigitalLifeTurn = d.digitalLife.onUserTurn({
      pad: s.currentPAD,
      memory: d.memorySystem,
      motivationState: d.motivationState,
      userText: cognitiveInput,
      userModel: d.userModelInst,
      mainEvent,
      selfModel: d.selfModel,
      relScore: effectiveRelScore(d.memorySystem.getRelationshipScore()),
      idleMs: idleMsSinceUser,
      rl: d.reinforcementLearning,
      previousBehaviorId: s.lastChatBehaviorId,
      externalTraits: d.personalityEvolution.traits,
      memoryAdmission,
    });
    if (s.lastDigitalLifeTurn?.padDelta) {
      const padDelta = s.lastDigitalLifeTurn.padDelta;
      if (padDelta.P || padDelta.A || padDelta.D) {
        s.currentPAD = updatePAD(s.currentPAD, padDelta, 0.2);
      }
    }

    // ══ ⑦ 用户理解（习惯提取等；不向模型注入「该怎样说话」的个性化脚本）══
    const analyticsResult = isRealUserTurn ? d.analyticsInst.analyze(cognitiveInput, {
      persistText: memoryAdmission.allowInference || memoryAdmission.allowProfile,
    }) : null;
    if (isRealUserTurn) d.habitExtractor.maybeRun(d.analyticsInst._log);

    // ── BDI 推断：每5轮异步触发，失败静默忽略，结果写入 UserModel ──
    const inferenceLines = recentUserLinesForMem.filter((line) => d.memoryAdmission.assessUserText(line, { source: 'user' }).allowInference);
    if (isRealUserTurn && memoryAdmission.allowInference && s.chatTurnCounter % 5 === 1 && inferenceLines.length > 0) {
      inferUserBdi({ recentUserLines: inferenceLines, timeoutMs: 6000 })
        .then(bdi => {
          if (bdi) {
            d.userModelInst.applyInferredBdi(bdi);
            console.log(`[bdi] 推断完成: beliefs=${bdi.beliefs.length} desires=${bdi.desires.length} intentions=${bdi.intentions.length}`);
          }
        })
        .catch(() => {});
    }

    if (s.chatTurnCounter % 10 === 0) {
      const pending = d.userModelInst.popConfirmation();
      if (pending) {
        d.goalSystem.goals.unshift({
          id: `CONFIRM_${pending.key}`,
          label: `确认推测：${pending.key}`,
          priority: 0.7,
          turns_remaining: 1,
          behavior_hint: '若自然可顺带一提',
          prompt_injection: `她心里有个想确认的点：${pending.question}（不必问卷式，接话时带过即可）`,
        });
      }
    }

    console.log(`[user_model] 用户特征: ${Object.entries(d.userModelInst.model.preferences).filter(([, v]) => v > 0.5).map(([k, v]) => `${k}:${v.toFixed(2)}`).join(' ')}`);

    // ══ RAG 与 动机/行为/策略/人格/元认知 并行（RAG 仅懒注入命中时）══
    const ragMs = Number(process.env.AMADEUS_RAG_MS) || 900;
    const ragQuery = autonomyInitiative ? (lastRealUserLine || userContent) : userContent;
    const ragPromise = useRagForTurn
      ? Promise.race([
          d.retrieveTopContexts(ragQuery, autonomyInitiative ? 2 : 3),
          new Promise((resolve) => setTimeout(() => resolve([]), ragMs)),
        ]).catch(() => [])
      : Promise.resolve([]);

    const statePromise = Promise.resolve().then(() => {
      const memBias = d.memorySystem.getLongTermPadBias();
      const relScore = effectiveRelScore(d.memorySystem.getRelationshipScore());
      s.lastRlStateKey = d.reinforcementLearning.buildStateKey(s.currentPAD, relScore);
      d.motivSystem.update(s.currentPAD, memBias, relScore);
      const motivSummary = d.motivSystem.getSummary();
      const behaviorResult = d.behaviorSys.decide(
        s.currentPAD, d.motivSystem, d.memorySystem, cognitiveInput, d.reinforcementLearning,
        {
          driveBoosts: s.lastDigitalLifeTurn?.driveBoosts || {},
          rlStateKey: s.lastRlStateKey,
        },
      );
      s.lastChatBehaviorId = behaviorResult.behaviorId;
      d.selfModel.update(
        s.currentPAD, memBias, relScore,
        behaviorResult.behaviorId,
        d.memorySystem.getRecentSignificant(3).join('; ')
      );
      const selfCtx = d.selfModel.toPromptContext();
      d.goalSystem.generateGoals(s.currentPAD, d.selfModel, relScore, d.memorySystem, d.digitalLife.autonomy.curiosity, {
        replyingToProactive,
      });
      if (s.lastDigitalLifeTurn?.goalSeeds?.length) {
        d.goalSystem.ingestUrgeGoals(s.lastDigitalLifeTurn.goalSeeds);
      }
      d.goalSystem.tick(behaviorResult.behaviorId, evDelta);
      const goalInjection = d.goalSystem.getActiveInjection();
      console.log(`[goal] 活跃目标: ${d.goalSystem.getSummary()}`);
      d.strategyLayer.evaluate(s.currentPAD, relScore, behaviorResult.behaviorId, d.goalSystem.goalHistory);
      const strategyCtx = d.strategyLayer.toPromptContext();
      console.log(`[strategy] 当前策略: ${d.strategyLayer.getLabel()}`);
      const recentEvent = { type: mainEvent.type || 'neutral' };
      d.personalityEvolution.updateTraits(recentEvent);
      d.personalityEvolution.updateValues(recentEvent);
      d.digitalLife.evolution.personality.ingestExternalTraits(d.personalityEvolution.traits);
      d.digitalLife.evolution.personality.updateFromEvent(recentEvent);
      const evolvedPersonalityLine = d.personalityEvolution.getDescription();
      console.log(`[personality] ${evolvedPersonalityLine}`);
      d.selfReflection.reflectOnDecision({
        action: behaviorResult.behaviorId,
        reasoning: behaviorResult.reasoning,
        factors: behaviorResult.reasons || [],
      });
      const dlMetacog = d.digitalLife.afterBehaviorDecision({
        mainEvent,
        decision: {
          action: behaviorResult.behaviorId,
          behaviorId: behaviorResult.behaviorId,
          reasoning: behaviorResult.reasoning,
          factors: behaviorResult.reasons || [],
        },
        chatTurnCounter: s.chatTurnCounter,
        chatMinimal: String(process.env.AMADEUS_CHAT_MINIMAL || '1').trim() !== '0',
      });
      const keywordConflicts = d.valueConsistency.detectConflicts({ description: userContent });
      if (keywordConflicts.length > 0) {
        console.log(`[metacognition] 价值观关键词冲突: ${keywordConflicts.map(c => c.description).join('; ')}`);
      }
      let latestInsight = dlMetacog?.insight?.content || '';
      const chatMinimal = String(process.env.AMADEUS_CHAT_MINIMAL || '1').trim() !== '0';
      if (!latestInsight && !chatMinimal && s.chatTurnCounter % 10 === 0) {
        const insight = d.selfReflection.generateInsight();
        if (insight) {
          latestInsight = insight.content;
          console.log(`[metacognition] 洞察: ${insight.content}`);
        }
      }
      if (latestInsight) {
        console.log(`[metacognition] 洞察: ${latestInsight}`);
        const injection = dlMetacog?.insightInjection
          || d.selfReflection.insightToGoalInjection({ content: latestInsight });
        if (injection) {
          d.goalSystem.goals.unshift({
            id: 'METACOG_INSIGHT',
            label: '元认知修正',
            priority: 0.55,
            turns_remaining: 2,
            behavior_hint: injection,
            prompt_injection: injection,
          });
        }
      }
      d.innerStateSix.updateFromTurn({
        pad: s.currentPAD,
        relScore,
        userEmotion: s.lastDigitalLifeTurn?.recognized?.emotion || d.analyzeUserEmotion(cognitiveInput),
        mainEvent,
        userText: cognitiveInput,
        behaviorId: behaviorResult.behaviorId,
      });
      let whoamiName = '';
      let whoamiSnippet = '';
      let whoamiForPresence = {};
      try {
        whoamiForPresence = ensureWhoamiOnDisk(d.whoamiPath);
        whoamiName = resolvePartnerDisplayName(whoamiForPresence) || '';
        const wp = [];
        if (whoamiName) wp.push(whoamiName);
        if (partnerIsOkabe(whoamiForPresence)) wp.push('冈部·很熟');
        if (whoamiForPresence.traits?.length) wp.push(whoamiForPresence.traits.slice(0, 3).join('、'));
        if (whoamiForPresence.relationship_note) wp.push(whoamiForPresence.relationship_note);
        if (wp.length) whoamiSnippet = wp.join('；');
      } catch (_) { /* ignore */ }
      const obsSummary = d.memorySystem.getObservationsSummary(2);
      const presence = derivePresence(
        s.currentPAD,
        cognitiveInput,
        behaviorResult.behaviorId,
        { closeness: Math.max(0, relScore), trust: 0.5 + relScore * 0.5 },
        { displayName: whoamiName, recentUserLines: recentUserLinesForMem.slice(-8) },
        {
          whoamiSnippet,
          obsSummary,
          idleMsSinceUser,
          isAutonomy: autonomyInitiative,
          replyingToProactive,
          proactiveAnchor,
          partnerIsOkabe: partnerIsOkabe(whoamiForPresence),
          lastUserAnchor: lastRealUserLine,
        },
      );
      return {
        memBias,
        relScore,
        motivSummary,
        behaviorResult,
        behaviorDirective: d.behaviorSys.toPromptConstraint(behaviorResult),
        presenceCtx: presenceToPromptLine(presence),
        turnStyleBlock: buildTurnStyleBlock({
          emotion: s.currentPAD,
          behaviorId: behaviorResult.behaviorId,
          behaviorLabel: behaviorResult.label,
          presence,
          closeness: Math.max(0, relScore),
          userText: cognitiveInput,
          partnerIsOkabe: partnerIsOkabe(whoamiForPresence),
        }),
        selfCtx,
        goalInjection,
        strategyCtx,
        personalityCtx: evolvedPersonalityLine,
        keywordConflicts,
        latestInsight,
        digitalLifeCtx: d.digitalLife.buildPromptContext({
          pad: s.currentPAD,
          memory: d.memorySystem,
          selfModel: d.selfModel,
          userModel: d.userModelInst,
          relScore,
          closeness: d.userModelInst.model?.relationship?.closeness ?? relScore,
          userText: cognitiveInput,
          idleMs: idleMsSinceUser,
          resonanceLine: s.lastDigitalLifeTurn?.resonanceLine || '',
          subtextLine: s.lastDigitalLifeTurn?.subtextLine || '',
          mentalModelLine: s.lastDigitalLifeTurn?.mentalModelLine || '',
          pendingNeed: s.lastDigitalLifeTurn?.pendingNeed || '',
          timeLine: s.lastDigitalLifeTurn?.timeLine || '',
          autonomyHint: autonomyDecision?.speakHint || s.lastDigitalLifeTurn?.autonomyPrompt || '',
          metacognitionInsight: latestInsight,
          includeDream: autonomyInitiative || idleMsSinceUser > 20 * 60 * 1000,
        }),
        expression: s.lastDigitalLifeTurn?.expression || d.digitalLife.embodiment.expression.snapshot(),
      };
    });

    const [ragHits, st] = await Promise.all([ragPromise, statePromise]);
    const ragAnchor = autonomyInitiative ? lastRealUserLine : userContent;
    let ragFiltered = filterRagHits(ragHits, ragAnchor, {});
    if (autonomyInitiative) {
      ragFiltered = filterAutonomyRagHits(ragFiltered, lastRealUserLine);
    }
    if (ragHits.length && ragFiltered.length < ragHits.length) {
      console.log(`[rag] 门控剔除 ${ragHits.length - ragFiltered.length} 条弱相关命中`);
    }
    const ragCtx = ragFiltered.length
      ? ragFiltered.map((h, i) => `(${i + 1}) [${h.source}] ${h.text}`).join('\n')
      : '';

    const userModelCtx = d.userModelInst.toPromptContext();

    const convHours = Number(process.env.AMADEUS_CONVERSATION_HOURS) || 14;
    const convChars = Number(process.env.AMADEUS_CONVERSATION_CHARS) || 1400;
    const convRecallChars = Number(process.env.AMADEUS_CONVERSATION_RECALL_CHARS) || 2600;
    const recallTurn = needsConversationRecall(userContent);
    const conversationCtx = d.unifiedDialogueLog.toPromptBlock({
      hours: convHours,
      sinceStartOfDay: true,
      maxChars: recallTurn ? convRecallChars : convChars,
      userText: userContent,
    });
    if (conversationCtx) {
      console.log(`[conversation] 注入实录 ${conversationCtx.length} 字${recallTurn ? '（核对/回忆加强）' : ''}`);
    }

    let memCtxCombined = '';
    if (useLongTermMemory) {
      const recentSig = d.memorySystem.getRecentSignificant(3);
      const memCtx = recentSig.length
        ? `【记忆碎片（高权重）】\n${recentSig.join('\n')}`
        : '';
      const todayTimeline = d.memorySystem.getTodayTimeline();
      const obsSummary = d.memorySystem.getObservationsSummary(3);
      const patterns = d.memorySystem.getPatterns(3);
      let timelineCtx = '';
      if (todayTimeline) timelineCtx += `【今日轨迹】\n${todayTimeline}\n`;
      if (obsSummary) timelineCtx += `【观察积累】\n${obsSummary}\n`;
      if (patterns.length) {
        timelineCtx += `【已发现模式】\n${patterns.map(p => `${p.label} (置信度:${p.confidence.toFixed(1)}) ${p.note || ''}`).join('\n')}\n`;
      }
      memCtxCombined = (timelineCtx.trim() && memCtx) ? `${timelineCtx.trim()}\n\n${memCtx}` : (timelineCtx.trim() || memCtx);
      if (autonomyInitiative && memCtxCombined) {
        memCtxCombined = filterAutonomyMemCtx(memCtxCombined, lastRealUserLine);
      }
    }

    const padDesc = `P:${s.currentPAD.P.toFixed(2)} A:${s.currentPAD.A.toFixed(2)} D:${s.currentPAD.D.toFixed(2)} S:${s.currentPAD.S.toFixed(2)}`;
    const relScore = effectiveRelScore(st.relScore);
    d.userModelInst.syncRelationshipFromScore(relScore);

    const closenessForCompanion = Math.max(0, relScore);
    const trustForCompanion = 0.5 + relScore * 0.5;
    let memSnippetForCompanion = '';
    if (useLongTermMemory) {
      const sig = d.memorySystem.getRecentSignificant(1);
      if (sig.length) memSnippetForCompanion = sig[0];
    }
    const companionBlock = buildCompanionBlock({
      P: s.currentPAD.P,
      A: s.currentPAD.A,
      closeness: closenessForCompanion,
      trust: trustForCompanion,
      userText: cognitiveInput,
      memSnippet: memSnippetForCompanion,
      userPresence: userPresenceState,
      isAutonomy: autonomyInitiative,
    });
    const relHigh = relScore > 0.38;

    const clientCtxRaw = normalizeClientContext(req.body.clientContext || {});
    const clientContextBlock = buildClientContextBlock({
      ...clientCtxRaw,
      wantLongMemory: clientCtxRaw.wantLongMemory || useLongTermMemory,
    });

    let whoamiCtx = '';
    let whoamiRecord = {};
    try {
      whoamiRecord = ensureWhoamiOnDisk(d.whoamiPath);
      const parts = [];
      const displayName = resolvePartnerDisplayName(whoamiRecord);
      if (displayName) {
        parts.push(
          partnerIsOkabe(whoamiRecord)
            ? `正在和 ${displayName}（冈部）对话——你们很熟，日常拌嘴，不是第一次见面。`
            : `正在和 ${displayName} 对话`,
        );
      }
      if (whoamiRecord.traits?.length) parts.push(`你对他的印象：${whoamiRecord.traits.join('、')}`);
      if (whoamiRecord.preferences?.length) parts.push(`他的喜好：${whoamiRecord.preferences.join('、')}`);
      if (whoamiRecord.basics && Object.keys(whoamiRecord.basics).length) {
        parts.push(`已知信息：${Object.entries(whoamiRecord.basics).map(([k, v]) => `${k}=${v}`).join('，')}`);
      }
      if (whoamiRecord.relationship_note) parts.push(whoamiRecord.relationship_note);
      if (parts.length) whoamiCtx = parts.join('\n');
    } catch (_) { /* ignore */ }

    const valueBlock = st.keywordConflicts?.length
      ? `【价值观拉扯】${_clipInnerPrompt(st.keywordConflicts.map((c) => c.description).join('；'), 140)}`
      : '';

    const lastAssistantBeforeTurn = [...dialogueForOllama].reverse().find((m) => m?.role === 'assistant')?.content || '';
    const utteranceFocus = utteranceFocusLine(userContent, {
      replyingToProactive,
      proactiveAnchor,
      lastAssistant: lastAssistantBeforeTurn,
    });
    const continuityFollowup = isContinuityFollowup(userContent, lastAssistantBeforeTurn);
    let continuityMessages = null;
    if (continuityFollowup) {
      const continuityModel = String(
        process.env.AMADEUS_CONTINUITY_MODEL
        || model,
      ).trim();
      if (continuityModel) model = continuityModel;
      temp = Math.min(Number(temp) || 0.72, 0.3);
      const previousUser = [...dialogueForOllama]
        .slice(0, -1)
        .reverse()
        .find((m) => m?.role === 'user')?.content || '';
      continuityMessages = buildContinuityRepairMessages({
        userText: userContent,
        lastAssistant: lastAssistantBeforeTurn,
        previousUser,
      });
    }
    if (continuityFollowup && dialogueForOllama.length) {
      const lastIndex = dialogueForOllama.length - 1;
      const lastMessage = dialogueForOllama[lastIndex];
      const previousAssistant = [...dialogueForOllama]
        .slice(0, lastIndex)
        .reverse()
        .find((m) => m?.role === 'assistant')?.content || '';
      if (lastMessage?.role === 'user') {
        dialogueForOllama[lastIndex] = {
          ...lastMessage,
          content: resolveFollowupForModel(lastMessage.content, previousAssistant),
        };
      }
    }
    const engagementHint = buildEngagementHint(d.userModelInst, userContent, relScore);
    const proactiveContinuity = replyingToProactive && proactiveAnchor
      ? buildProactiveReplyFocus(userContent, proactiveAnchor)
      : '';
    const autonomyContinuity = autonomyInitiative
      ? buildAutonomyContinuityBlock({
          lastUserText: lastRealUserLine,
          userPresence: userPresenceState,
          lastKurisuLine: (() => {
            for (let i = parsed.dialogue.length - 1; i >= 0; i--) {
              const m = parsed.dialogue[i];
              if (m && m.role === 'assistant') return String(m.content || '').trim();
            }
            return '';
          })(),
        })
      : '';

    const behaviorContext = {
      soulContent: systemContent,
      voiceContent: d.cachedVoiceContent,
      replyLanguage,
      useLongTermMemory,
      utteranceFocus,
      proactiveContinuity,
      autonomyContinuity,
      replyingToProactive,
      autonomyInitiative,
      lastRealUserLine,
      proactiveAnchor,
      userPresence: userPresenceState,
      engagementHint,
      emotion: { P: s.currentPAD.P, A: s.currentPAD.A, D: s.currentPAD.D, S: s.currentPAD.S },
      relationship: {
        closeness: Math.max(0, relScore),
        trust: 0.5 + relScore * 0.5,
      },
      motivation: d.motivationState,
      userProfile: whoamiCtx,
      userModelCtx: userModelCtx ? `【用户理解】\n${userModelCtx}` : '',
      motivSummary: st.motivSummary || '',
      selfCtx: st.selfCtx || '',
      behaviorDirective: st.behaviorDirective || '',
      presenceCtx: st.presenceCtx || '',
      turnStyleBlock: st.turnStyleBlock || '',
      companionBlock,
      latestInsight: st.latestInsight || '',
      digitalLifeCtx: st.digitalLifeCtx || '',
      personalityCtx: st.personalityCtx || '',
      valueBlock,
      innerStateSixBlock: d.innerStateSix.toPromptBlock(),
      socialIdentityBlock: buildSocialIdentityPrompt({
        userText: cognitiveInput,
        pad: s.currentPAD,
        relScore,
        relHigh,
        recentEvents: d.memorySystem.events.slice(-6),
      }),
      expressionVariantBlock: '',
      emotionalBandwidthBlock: '',
      behaviorContextLine: d.behaviorIngest.toPromptLine(),
      clientContextBlock,
      ragCtx: useLongTermMemory && ragCtx ? `【背景知识】\n${ragCtx}` : '',
      conversationCtx,
      conversationRecall: recallTurn,
      memCtx: memCtxCombined,
      strategyContext: useLongTermMemory ? st.strategyCtx : '',
      goalInjection: useLongTermMemory ? st.goalInjection : '',
    };

    const affect = (d.emotionalBandwidth || new EmotionalBandwidthEngine()).resolve({
      userText: cognitiveInput,
      pad: s.currentPAD,
      relScore,
      relHigh,
    });
    behaviorContext.emotionalBandwidthBlock = affect.block || '';
    behaviorContext._affectBand = affect.band || '';
    behaviorContext.expressionVariantBlock = buildExpressionVariantBlock(
      s.currentPAD,
      d.innerStateSix,
      { relHigh, affectBand: affect.band || '' },
    );

    behaviorContext.recentUserLines = recentUserLinesForMem.slice(-8);
    behaviorContext.partnerIsOkabe = partnerIsOkabe(whoamiRecord);
    behaviorContext.displayName = resolvePartnerDisplayName(whoamiRecord) || behaviorContext.displayName || '';
    behaviorContext.partnerCtx = buildPartnerContextBlock(whoamiRecord, cognitiveInput);

    if (d.brainPipeline) {
      const enriched = d.brainPipeline.beforePrompt({
        perceived: {
          parsed,
          userContent,
          autonomyInitiative,
          idleMsSinceUser,
          replyingToProactive,
          proactiveAnchor,
          recentUserLinesForMem,
          lastRealUserLine,
          userPresenceState,
          useLongTermMemory,
          useRagForTurn,
          cognitiveInput,
        },
        reqBody: req.body,
        digitalLifeTurn: s.lastDigitalLifeTurn,
        pad: s.currentPAD,
        behaviorContext,
      });
      if (enriched.brainSlimMode || enriched.brainWorkspaceBlock) {
        behaviorContext.brainSlimMode = true;
        behaviorContext.brainWorldSummary = enriched.brainWorldSummary;
        behaviorContext.brainSelfSummary = enriched.brainSelfSummary;
        behaviorContext.brainDeliberationBlock = enriched.brainDeliberationBlock;
        behaviorContext.brainWorkspaceBlock = enriched.brainWorkspaceBlock || '';
        behaviorContext.digitalLifeCtx = enriched.digitalLifeCtx ?? '';
        behaviorContext.skipSymbolicInPrompt = enriched.skipSymbolicInPrompt;
      }
    }

    const symbolicRules = behaviorContext.skipSymbolicInPrompt
      ? []
      : symbolicReasoning(cognitiveInput, s.currentPAD, behaviorContext);
    if (symbolicRules.length > 0) {
      console.log(`[symbolic] 触发规则: ${symbolicRules.map(r => r.reason).join(', ')}`);
    }

    const systemPrompt = buildPrompt(behaviorContext, symbolicRules);

    console.log(`[chat] PAD=${padDesc} rel=${relScore.toFixed(2)} events=${d.memorySystem.events.length} behavior=${st.behaviorResult.label}`);
    const fullPrompt = systemPrompt;
    const envPromptCap = Number(process.env.AMADEUS_MAX_PROMPT_CHARS);
    const numCtxEnv = Number(process.env.AMADEUS_OLLAMA_NUM_CTX);
    const standardNumCtx = Number.isFinite(numCtxEnv) && numCtxEnv > 0 ? numCtxEnv : 2048;
    const fastCtxEnv = Number(process.env.AMADEUS_FAST_CHAT_NUM_CTX);
    const numCtx = fastConversation
      ? (Number.isFinite(fastCtxEnv) && fastCtxEnv >= 768 ? fastCtxEnv : 1024)
      : standardNumCtx;
    let maxPromptChars = resolvePromptCharBudget({
      numCtx,
      maxTok,
      envMaxChars: Number.isFinite(envPromptCap) && envPromptCap > 0 ? envPromptCap : 8000,
    });
    if (envPromptCap > maxPromptChars) {
      console.warn(`[chat] AMADEUS_MAX_PROMPT_CHARS=${envPromptCap} 超出 num_ctx=${numCtx} 安全预算，已压至 ${maxPromptChars} 字`);
    }
    if (fastConversation) {
      const fastPromptEnv = Number(process.env.AMADEUS_FAST_CHAT_PROMPT_CHARS);
      const fastPromptCap = Number.isFinite(fastPromptEnv) && fastPromptEnv >= 900 ? fastPromptEnv : 1500;
      maxPromptChars = Math.min(maxPromptChars, fastPromptCap);
    }
    const repPen = Number(process.env.AMADEUS_OLLAMA_REPEAT_PENALTY);
    const ollamaOptions = {
      temperature: temp,
      num_predict: maxTok,
      repeat_penalty: Number.isFinite(repPen) && repPen > 0 ? repPen : 1.12,
      num_ctx: numCtx,
    };
    const keepAlive = String(process.env.AMADEUS_OLLAMA_KEEP_ALIVE || '2m').trim() || '2m';

    // #region agent log
    d.agentDebugLog({ hypothesisId: 'A-D', location: 'server.js:chat.preOllama', message: 'ollama request shape', data: { model, useStream, fastConversation, fullPromptLen: fullPrompt.length, maxPromptChars, maxTok, numCtx: ollamaOptions.num_ctx, num_predict: ollamaOptions.num_predict, repeat_penalty: ollamaOptions.repeat_penalty, keepAlive } });
    // #endregion

    // ══ 调用 Ollama（500/503 时自动缩短 prompt 重试，减轻显存/上下文压力）══
    const ollamaStartTime = Date.now();
    let ollamaRes;
    let responseProvider = 'ollama';
    const initialPromptForModel = fitSystemForDialogue(fullPrompt, dialogueForOllama, maxPromptChars);
    const initialMessages = continuityMessages?.length
      ? continuityMessages
      : buildOllamaMessages(initialPromptForModel, dialogueForOllama, maxPromptChars);
    if (typeof d.requestWorkBrainChat === 'function') {
      try {
        const remote = await d.requestWorkBrainChat(initialMessages, {
          stream: useStream,
          temperature: temp,
          maxTokens: maxTok,
          signal: lease.signal,
        });
        if (remote?.response) {
          ollamaRes = remote.response;
          responseProvider = remote.provider || 'work-brain';
          console.log(`[chat] ${responseProvider} responded with ${remote.model || 'configured model'}`);
        }
      } catch (e) {
        console.warn(`[chat] work brain unavailable, falling back to Ollama: ${e.message}`);
      }
    }

    for (let attempt = 0; !ollamaRes && attempt < 3; attempt++) {
      const promptForModel = fitSystemForDialogue(fullPrompt, dialogueForOllama, maxPromptChars);
      const ollamaMessages = continuityMessages?.length
        ? continuityMessages
        : buildOllamaMessages(promptForModel, dialogueForOllama, maxPromptChars);
      const msgChars = estimateMessageChars(ollamaMessages);
      if (fullPrompt.length + msgChars > maxPromptChars) {
        console.warn(`[chat] system ${fullPrompt.length} + dialogue ~${msgChars} > ${maxPromptChars}, 已压缩 system 并保留 ${Math.max(0, ollamaMessages.length - 1)} 轮对话`);
      }
      console.log(`[chat] Prompt system=${promptForModel.length} msgs=${ollamaMessages.length} ~chars=${msgChars}, Model: ${model}, try=${attempt + 1}`);
      // #region agent log
      d.agentDebugLog({ hypothesisId: 'A-D', location: 'server.js:chat.ollamaAttempt', message: 'before fetch', data: { attempt: attempt + 1, model, promptForModelLen: promptForModel.length, ollamaMsgCount: ollamaMessages.length, msgChars, maxPromptCharsCap: maxPromptChars, stream: useStream } });
      // #endregion
      ollamaRes = await fetch(`${d.OLLAMA_BASE}/api/chat`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ 
          model, 
          messages: ollamaMessages,
          stream:useStream,
          think:false,
          options: ollamaOptions,
          keep_alive: keepAlive,
        }),
        signal: lease.signal,
      });
      if (ollamaRes.ok) {
        console.log(`[chat] Ollama responded in ${Date.now() - ollamaStartTime}ms`);
        // #region agent log
        d.agentDebugLog({ hypothesisId: 'A-D', location: 'server.js:chat.ollamaOk', message: 'ollama ok', data: { attempt: attempt + 1, model, ms: Date.now() - ollamaStartTime } });
        // #endregion
        break;
      }
      const detail = await d.readOllamaErrorBody(ollamaRes);
      const looksLikeModelLoadFail = d.ollamaErrorLooksLikeModelLoadFail(detail);
      const looksLikeContextOverflow = d.ollamaErrorLooksLikeContextOverflow(detail);
      const willShortenRetry = attempt < 2 && !looksLikeModelLoadFail && (
        [500, 503].includes(ollamaRes.status) ||
        (ollamaRes.status === 400 && looksLikeContextOverflow)
      );
      // #region agent log
      d.agentDebugLog({ hypothesisId: 'B', location: 'server.js:chat.ollamaErr', message: 'ollama non-ok', data: { attempt: attempt + 1, model, httpStatus: ollamaRes.status, detailSlice: String(detail).slice(0, 220), looksLikeModelLoadFail, looksLikeContextOverflow, willShortenRetry } });
      // #endregion
      if (willShortenRetry) {
        maxPromptChars = attempt === 0
          ? Math.max(1200, Math.floor(maxPromptChars * 0.72))
          : Math.max(900, Math.floor(maxPromptChars * 0.55));
        console.warn(`[chat] Ollama ${ollamaRes.status}, 缩短上下文重试 cap=${maxPromptChars}`, detail.slice(0, 160));
        continue;
      }
      const loadHint = looksLikeModelLoadFail
        ? ' （模型加载/资源问题通常与 prompt 长度无关：检查显存、`ollama ps`、其它占 GPU 进程，或换更小模型。）'
        : '';
      throw new Error((detail ? `Ollama ${ollamaRes.status}: ${detail}` : `Ollama HTTP ${ollamaRes.status}`) + loadHint);
    }
    if (!ollamaRes.ok) {
      throw new Error('Ollama 多次重试仍失败');
    }

    // 非流式（/api/chat 返回 message.content；/api/generate 才是 response）
    if (!useStream) {
      const extractChatText = (payload) => {
        const msg = payload?.message || {};
        return String(
          msg.content
          || payload?.response
          || payload?.choices?.[0]?.message?.content
          || ''
        );
      };
      let ollamaPayload = await ollamaRes.json();
      let raw = extractChatText(ollamaPayload);
      let content = d.stripModelThinkingAll(raw);
      // content 空但 thinking 里有正文时再剥一次；仍空则加长重试一轮
      if (!String(content || '').trim()) {
        const thinkingBits = [ollamaPayload?.message?.thinking, ollamaPayload?.thinking]
          .filter(Boolean).join('\n');
        if (thinkingBits) content = d.stripModelThinkingAll(thinkingBits);
      }
      if (!String(content || '').trim() && Number(ollamaOptions.num_predict) < 512) {
        console.warn('[chat] empty content after first pass, retry with higher num_predict');
        const retryOpts = { ...ollamaOptions, num_predict: 512 };
        const promptForModel = fitSystemForDialogue(fullPrompt, dialogueForOllama, maxPromptChars);
        const ollamaMessages = continuityMessages?.length
          ? continuityMessages
          : buildOllamaMessages(promptForModel, dialogueForOllama, maxPromptChars);
        const retryRes = await fetch(`${d.OLLAMA_BASE}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            messages: ollamaMessages,
            stream: false,
            think: false,
            options: retryOpts,
            keep_alive: keepAlive,
          }),
          signal: lease.signal,
        });
        if (retryRes.ok) {
          ollamaPayload = await retryRes.json();
          raw = extractChatText(ollamaPayload);
          content = d.stripModelThinkingAll(raw);
          if (!String(content || '').trim()) {
            const thinkingBits = [ollamaPayload?.message?.thinking, ollamaPayload?.thinking]
              .filter(Boolean).join('\n');
            if (thinkingBits) content = d.stripModelThinkingAll(thinkingBits);
          }
        }
      }
      content = stripChatMarkdown(content);
      {
        const accepted = await _acceptAgencyTaskFromReply(d, content, { turnId: lease.turnId });
        content = accepted.spoken || content;
      }
      const userCorpus = recentUserLinesForMem.join('\n');
      content = stripOrphanClosingSentence(content, userContent, userCorpus);
      const oocOpts = autonomyInitiative
        ? { autonomy: true, userAnchor: lastRealUserLine }
        : {};
      if (d.brainPipeline?.processReply) {
        content = await d.brainPipeline.processReply(content, { userText: userContent, oocOpts });
      } else {
        content = d.applyOocRepair(content, userContent, '', oocOpts);
      }
       if (!lease.isCurrent()) return;
       const polished = await d.postReplyPadUpdate(content, userContent, {
         situation: clientCtxRaw.situation,
         autonomy: autonomyInitiative,
         conversationId: lease.conversationId,
         turnId: lease.turnId,
       });
       const out = polished.chinese || content;
       if (!lease.isCurrent()) return;
       if (out && behaviorContext._affectBand && d.emotionalBandwidth?.registerSpoken) {
         d.emotionalBandwidth.registerSpoken(behaviorContext._affectBand);
       }
       const sent = res.json({
         response: out,
         affectBand: behaviorContext._affectBand || null,
         choices:[{index:0,finish_reason:'stop',message:{role:'assistant',content: out}}],
       });
       lease.complete();
       return sent;
    }

    // 流式SSE
    res.setHeader('Content-Type','text/event-stream');
    res.setHeader('Cache-Control','no-cache');
    res.setHeader('Connection','keep-alive');
    res.flushHeaders();

    const reader  = ollamaRes.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf='', isThinking=false, fullResponse='', replyUpdated=false;
    let ollamaPieceCarry = { accum: '' };

    let streamFinalizePromise = null;
    const finalizeStream = () => {
      if (streamFinalizePromise) return streamFinalizePromise;
      streamFinalizePromise = (async () => {
        if (replyUpdated) return;
        replyUpdated = true;
        // 流式首轮空正文：常见于 thinking 吃光 num_predict；非流式加长重试一次
        if (!String(fullResponse || '').trim() && lease.isCurrent() && !res.writableEnded) {
          try {
            console.warn('[chat] empty stream content, retry non-stream with higher num_predict');
            const promptForModel = fitSystemForDialogue(fullPrompt, dialogueForOllama, maxPromptChars);
            const ollamaMessages = continuityMessages?.length
              ? continuityMessages
              : buildOllamaMessages(promptForModel, dialogueForOllama, maxPromptChars);
            const retryRes = await fetch(`${d.OLLAMA_BASE}/api/chat`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                model,
                messages: ollamaMessages,
                stream: false,
                think: false,
                options: { ...ollamaOptions, num_predict: Math.max(512, Number(ollamaOptions.num_predict) || 0) },
                keep_alive: keepAlive,
              }),
              signal: lease.signal,
            });
            if (retryRes.ok) {
              const payload = await retryRes.json();
              let recovered = d.stripModelThinkingAll(
                payload?.message?.content || payload?.response || ''
              );
              if (!recovered) {
                recovered = d.stripModelThinkingAll(
                  [payload?.message?.thinking, payload?.thinking].filter(Boolean).join('\n')
                );
              }
              recovered = stripChatMarkdown(recovered);
              if (recovered) {
                fullResponse = recovered;
                res.write(`data: ${JSON.stringify({ text: recovered })}\n\n`);
              }
            }
          } catch (e) {
            console.warn('[chat] empty-stream retry failed:', e.message);
          }
        }
        const userCorpus = recentUserLinesForMem.join('\n');
        const cleanedRaw = stripRoleplayActions(stripChatMarkdown(d.stripModelThinkingAll(fullResponse)));
        let cleaned = cleanedRaw;
        {
          const accepted = await _acceptAgencyTaskFromReply(d, cleanedRaw, { turnId: lease.turnId });
          cleaned = accepted.spoken || cleanedRaw;
        }
        let trimmed = stripOrphanClosingSentence(cleaned, userContent, userCorpus);
        const oocOpts = autonomyInitiative
          ? { autonomy: true, userAnchor: lastRealUserLine }
          : {};
        const canonicalStreamText = trimmed || cleaned;
        // 日语模式下禁止跳过定稿：否则实录存原文、界面另译，两边必然分叉且会通顺化脑补
        const mustPolish = getReplyLanguageMode() === 'ja'
          || process.env.AMADEUS_JP_FIRST === '1';
        if (process.env.AMADEUS_FAST_STREAMING_VOICE !== '0' && !mustPolish) {
          if (!lease.isCurrent()) return;
          fullResponse = canonicalStreamText;
          await d.postReplyPadUpdate(canonicalStreamText, userContent, {
            situation: clientCtxRaw.situation,
            autonomy: autonomyInitiative,
            skipPolish: true,
            conversationId: lease.conversationId,
            turnId: lease.turnId,
          });
          if (canonicalStreamText && behaviorContext._affectBand && d.emotionalBandwidth?.registerSpoken) {
            d.emotionalBandwidth.registerSpoken(behaviorContext._affectBand);
          }
          if (lease.isCurrent() && !res.writableEnded) {
            res.write('data: [DONE]\n\n');
            res.end();
          }
          lease.complete();
          return;
        }
        if (d.brainPipeline?.processReply) {
          trimmed = await d.brainPipeline.processReply(trimmed, {
            userText: userContent,
            oocOpts,
            streamedRaw: cleaned,
          });
        } else {
          trimmed = stripRoleplayActions(d.applyOocRepair(trimmed, userContent, cleaned, oocOpts));
        }
        const oocFixed = trimmed || cleaned;
        fullResponse = oocFixed;
        try {
          if (!lease.isCurrent()) return;
          const polished = await d.postReplyPadUpdate(oocFixed, userContent, {
            situation: clientCtxRaw.situation,
            autonomy: autonomyInitiative,
            conversationId: lease.conversationId,
            turnId: lease.turnId,
          });
          const finalCn = polished.chinese || oocFixed;
          const finalJp = polished.japanese || '';
          if (finalCn) fullResponse = finalCn;
          if (finalCn && behaviorContext._affectBand && d.emotionalBandwidth?.registerSpoken) {
            d.emotionalBandwidth.registerSpoken(behaviorContext._affectBand);
          }
          const shouldReplace = (finalCn.length > 0)
            && (shouldReplaceStreamText(cleaned, finalCn)
              || finalCn !== cleaned
              || process.env.AMADEUS_JP_FIRST === '1');
          if (shouldReplace) {
            const payload = { replaceText: finalCn };
            if (finalJp) payload.modelJp = finalJp;
            res.write(`data: ${JSON.stringify(payload)}\n\n`);
          }
        } catch (e) {
          console.warn('[chat] stream finalize polish', e.message);
          if (shouldReplaceStreamText(cleaned, oocFixed) && oocFixed.length > 0) {
            res.write(`data: ${JSON.stringify({ replaceText: oocFixed })}\n\n`);
          }
        }
        if (lease.isCurrent() && !res.writableEnded) {
          res.write('data: [DONE]\n\n');
          res.end();
        }
        lease.complete();
      })();
      return streamFinalizePromise;
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream:true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        let ln = line.trim();
        if (!ln) continue;
        if (ln.startsWith('data:')) ln = ln.slice(5).trim();
        if (ln === '[DONE]') continue;
        try {
          const obj = JSON.parse(ln);
          const rawPiece = d.ollamaChatStreamRawPiece(obj);
          const { delta, carry } = d.ollamaStreamToDelta(rawPiece, ollamaPieceCarry);
          ollamaPieceCarry = carry;
          let token = delta;
          if (token.includes('\u003credacted_thinking\u003e')) { isThinking = true; token = token.split('\u003credacted_thinking\u003e').slice(-1)[0] || ''; }
          if (token.includes('\u003c\/redacted_thinking\u003e')) { isThinking = false; token = token.split('\u003c\/redacted_thinking\u003e').slice(-1)[0] || ''; }
          if (token.includes('\u003cthink\u003e')) { isThinking = true; token = token.split('\u003cthink\u003e').slice(-1)[0] || ''; }
          if (token.includes('\u003c\/think\u003e')) { isThinking = false; token = token.split('\u003c\/think\u003e').slice(-1)[0] || ''; }
          if (isThinking) continue;
          if (token) {
            fullResponse += token;
            res.write(`data: ${JSON.stringify({ text:token })}\n\n`);
          }
          if (obj.done || obj.choices?.[0]?.finish_reason === 'stop') {
            await finalizeStream();
            return;
          }
        } catch {}
      }
    }
    await finalizeStream();

  } catch (err) {
    const cancelled = lease.signal.aborted || err?.name === 'AbortError' || /aborted|superseded|client-disconnected/i.test(String(err?.message || ''));
    if (!cancelled) console.error('[chat]', err.stack || err.message);
    // #region agent log
    d.agentDebugLog({ hypothesisId: 'E', location: 'server.js:chat.catch', message: 'chat handler error', data: { errMsg: String(err && err.message || err).slice(0, 400), streamReq: __streamRequested } });
    // #endregion
    try {
      if (cancelled) {
        try {
          if (!res.writableEnded) res.end();
        } catch (_) { /* ignore */ }
        lease.complete();
        return;
      }
      if (!res.headersSent) {
        // 尚未开始写 SSE 时统一返回 JSON，便于 fetch 用 res.json() 读 error（避免 502+text/event-stream 混用）
        res.status(502).json({
          error: String(err.message),
          response: '',
          choices: [{ index: 0, finish_reason: 'error', message: { role: 'assistant', content: '' } }],
        });
      } else if (__streamRequested && !res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: String(err.message) })}\n\n`);
        res.write('data: [DONE]\n\n');
      }
    } catch (_) { /* ignore */ }
    try {
      if (!res.writableEnded) res.end();
    } catch (_) { /* ignore */ }
    lease.complete();
  }
}

module.exports = { init, runChatTurn };
