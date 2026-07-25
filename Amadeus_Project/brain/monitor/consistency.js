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

  if (/刚才那句不算|我重新说|说重点。?$|听着呢。?$|又是[这那]个话题吗|停止指定话题/.test(reply)) {
    violations.push({
      id: 'consistency.product_meta',
      severity: 'block',
      rewriteHint: '禁止产品元话语与语气词池；只说有思考的当场对白。',
    });
  }

  if (opts.autonomy && typeof turnContinuity.replyLooksLikeAutonomyFabrication === 'function') {
    if (turnContinuity.replyLooksLikeAutonomyFabrication(opts.userAnchor || userText, reply, {
      alreadyTalking: opts.alreadyTalking === true || !!String(opts.userAnchor || '').trim(),
    })) {
      violations.push({
        id: 'consistency.autonomy_fabrication',
        severity: 'block',
        rewriteHint: '主动轮禁止编造实验/电话/刚注意到；情景必须符合当前窗口。',
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
