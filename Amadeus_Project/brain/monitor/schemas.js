'use strict';

/**
 * @typedef {object} MonitorViolation
 * @property {string} id
 * @property {string} severity - 'block' | 'warn'
 * @property {string} rewriteHint
 */

/**
 * @typedef {object} MonitorResult
 * @property {boolean} pass
 * @property {number} confidence
 * @property {MonitorViolation[]} violations
 * @property {object[]} speechActs
 */

module.exports = {};
