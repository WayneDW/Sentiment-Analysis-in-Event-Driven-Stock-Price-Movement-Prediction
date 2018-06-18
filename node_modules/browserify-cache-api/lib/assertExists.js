var assert = require('assert');

function assertExists(value, name) {
  assert(value != null, 'missing ' + (name || 'argument'));
}

module.exports = assertExists;
