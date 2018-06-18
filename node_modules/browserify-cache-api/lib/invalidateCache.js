var assertExists = require('./assertExists');
var invalidateModifiedFiles = require('./invalidateModifiedFiles');

function invalidateCache(mtimes, cache, done) {
  assertExists(mtimes);

  invalidateModifiedFiles(mtimes, Object.keys(cache), function(file) {
    delete cache[file];
  }, done);
}

module.exports = invalidateCache;
