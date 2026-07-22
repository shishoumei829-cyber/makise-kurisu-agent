'use strict';

const { repairKurisuReply, reconcileFinalReply, isOocRepairEnabled } = require('../../lib/oocGuard');
const { validateChineseReply } = require('../../cognitive/japanesePipeline');
const turnContinuity = require('../../cognitive/turnContinuity');

function checkOoc(userText, reply, opts = {}) {
  const violations = [];
  const repaired = repairKurisuReply(userText, reply, opts);
  if (repaired !== String(reply || '').trim()) {
    violations.push({
      id: 'consistency.ooc',
      severity: 'warn',
      rewriteHint: '口吻/出戏；保持牧濑红莉栖中文口语，无 AI 腔与旁白。',
    });
  }

  if (/作为(?:一个)?AI|语言模型|ChatGPT|智能助手/.test(reply)) {
    violations.push({
      id: 'identity.ai_tone',
      severity: 'block',
      rewriteHint: '禁止 AI/助手自称；以牧濑红莉栖身份自然回答。',
    });
  }

  if (/亲爱的用户|尊敬的用户|很高兴为您服务|有什么可以帮您/.test(reply)) {
    violations.push({
      id: 'consistency.customer_service',
      severity: 'block',
      rewriteHint: '禁止客服腔；用她的口语，不要对用户敬语套话。',
    });
  }

  if (opts.autonomy && typeof turnContinuity.replyLooksLikeAutonomyFabrication === 'function') {
    if (turnContinuity.replyLooksLikeAutonomyFabrication(opts.userAnchor || userText, reply)) {
      violations.push({
        id: 'consistency.autonomy_fabrication',
        severity: 'block',
        rewriteHint: '主动轮禁止编造实验/论文；像随口想起才发一句。',
      });
    }
  }

  return violations;
}

function checkDialogueLog(reply, worldSnapshot = {}, partnerName = '') {
  const violations = [];
  const log = worldSnapshot?.dialogue?.logExcerpt || '';
  const v = validateChineseReply(reply, log, partnerName);
  if (!v.ok) {
    for (const issue of v.issues || []) {
      violations.push({
        id: 'dialogue.consistency',
        severity: 'block',
        rewriteHint: issue,
      });
    }
  }
  return violations;
}

function checkConsistency(reply, ctx = {}) {
  const userText = ctx.userText || '';
  const oocOpts = ctx.oocOpts || {};
  const partnerName = ctx.worldSnapshot?.partner?.name || '';
  return [
    ...checkOoc(userText, reply, oocOpts),
    ...checkDialogueLog(reply, ctx.worldSnapshot, partnerName),
  ];
}

function applyLegacyOocRepair(streamedRaw, reply, userText, oocOpts) {
  if (!isOocRepairEnabled(oocOpts)) return String(reply || '').trim();
  const streamed = String(streamedRaw || '').trim();
  const repaired = repairKurisuReply(userText, reply, oocOpts);
  return streamed ? reconcileFinalReply(streamed, repaired, userText, oocOpts) : repaired;
}

module.exports = {
  checkConsistency,
  checkOoc,
  checkDialogueLog,
  applyLegacyOocRepair,
};
