function Cache(cacheData) {
  var cache = cacheData || {};

  cache.modules = cache.modules || {}; // module-deps opt 'cache'
  cache.mtimes = cache.mtimes || {}; // maps cached file filepath to mtime when cached
  cache.dependentFiles = cache.dependentFiles || {};

  return cache;
}

module.exports = Cache;
