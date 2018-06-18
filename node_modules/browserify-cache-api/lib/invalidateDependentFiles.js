var invalidateModifiedFiles = require('./invalidateModifiedFiles');

function invalidateDependentFiles(cache, invalidatedModules, done) {
  var dependentFiles = cache.dependentFiles;

  // clean up maybe-no-longer-dependent modules
  var maybeNoLongerDependentModules = {};
  invalidatedModules.forEach(function(module) {
    maybeNoLongerDependentModules[module] = true;
  });
  Object.keys(dependentFiles).forEach(function(dependentFile) {
    if (dependentFiles[dependentFile]) {
      Object.keys(dependentFiles[dependentFile]).forEach(function(module) {
        if (maybeNoLongerDependentModules[module]) {
          delete dependentFiles[dependentFile][module];
        }
      });
    }
  });

  invalidateModifiedFiles(cache.mtimes, Object.keys(dependentFiles), function(dependentFile) {
    Object.keys(dependentFiles[dependentFile]).forEach(function(module) {
      delete cache.modules[module];
    });
  }, function(err) { done(err); });
}

module.exports = invalidateDependentFiles;
