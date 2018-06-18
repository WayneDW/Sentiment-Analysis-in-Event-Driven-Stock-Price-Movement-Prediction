var fs = require('fs');
var assert = require('assert');
var through = require('through2');
var assign = require('xtend/mutable');

var assertExists = require('./assertExists');
var proxyEvent = require('./proxyEvent');
var Cache = require('./Cache');
var invalidateCache = require('./invalidateCache');
var invalidateDependentFiles = require('./invalidateDependentFiles');

function BrowserifyCache(b, opts) {
  assertExists(b);
  opts = opts || {};

  if (BrowserifyCache.getCache(b)) return b; // already attached

  // certain opts must have been set when browserify instance was created
  assert(b._options.cache, "required browserify 'cache' opt not set");

  // load cache from file specified by cacheFile opt
  var cacheFile = opts.cacheFile || opts.cachefile || b._options && b._options.cacheFile || null;
  var cacheData = loadCacheData(b, cacheFile);

  // b._options.cache is a shared object into which loaded module cache is merged.
  // it will be reused for each build, and mutated when the cache is invalidated.
  assign(b._options.cache, cacheData.modules);
  cacheData.modules = b._options.cache;

  var cache = Cache(cacheData);
  BrowserifyCache.setCache(b, cache);

  attachCacheHooksToPipeline(b);
  attachCacheDiscoveryHandlers(b);
  attachCachePersistHandler(b, cacheFile);

  return b;
}

BrowserifyCache.args = {
  cache: {},
  packageCache: {},
};

BrowserifyCache.getCache = function(b) {
  return b.__cacheObjects;
};

BrowserifyCache.setCache = function(b, cache) {
  b.__cacheObjects = cache;
};

// keep track of deps which are pending for the purpose of writing cache file
// (eg. being transformed)
function addCacheBlocker(b) {
  if (b.__cacheBlockerCount == null) {
    b.__cacheBlockerCount = 0;
  }

  b.__cacheBlockerCount++;
}

function removeCacheBlocker(b) {
  assert(b.__cacheBlockerCount >= 1);

  b.__cacheBlockerCount--;

  if (b.__cacheBlockerCount === 0) {
    b.emit('_cacheReadyToWrite');
  }
}

function attachCacheHooksToPipeline(b) {
  var prevBundle = b.bundle;
  b.bundle = function(cb) {
    var outputStream = through.obj();

    invalidateCacheBeforeBundling(b, function(err) {
      if (err) return outputStream.emit('error', err);

      var bundleStream = prevBundle.call(b, cb);
      proxyEvent(bundleStream, outputStream, 'file');
      proxyEvent(bundleStream, outputStream, 'package');
      proxyEvent(bundleStream, outputStream, 'transform');
      proxyEvent(bundleStream, outputStream, 'error');
      bundleStream.pipe(outputStream);
    });

    return outputStream;
  };
}

function invalidateCacheBeforeBundling(b, done) {
  var cache = BrowserifyCache.getCache(b);

  invalidateCache(cache.mtimes, cache.modules, function(err, invalidated, deleted) {
    invalidateDependentFiles(cache, [].concat(invalidated, deleted), function(err) {
      b.emit('changedDeps', invalidated, deleted);
      done(err, invalidated);
    });
  });
}

function attachCacheDiscoveryHandlers(b) {
  // based on how watchify adds deps to cache
  function insertDepCollector() {
    b.pipeline.get('deps').push(through.obj(function(row, enc, next) {
      var file = row.expose ? b._expose[row.id] : row.file;

      var dep = {
        file: file,
        source: row.source,
        deps: assign({}, row.deps),
      };

      addCacheBlocker(b);
      updateCacheOnDep(b, dep, function(err) {
        if (err) b.emit('error', err);
        removeCacheBlocker(b);
      });

      this.push(row);
      next();
    }));
  }

  b.on('reset', insertDepCollector);
  insertDepCollector();

  b.on('transform', function(transformStream, moduleFile) {
    transformStream.on('file', function(dependentFile) {
      addCacheBlocker(b);
      updateCacheOnTransformFile(b, moduleFile, dependentFile, function(err) {
        if (err) b.emit('error', err);
        removeCacheBlocker(b);
      });
    });
  });
}

function updateCacheOnDep(b, dep, done) {
  var cache = BrowserifyCache.getCache(b);

  var file = dep.file || dep.id;
  if (typeof file === 'string') {
    if (dep.source != null) {
      cache.modules[file] = dep;
      if (!cache.mtimes[file])
        return updateMtime(cache.mtimes, file, done);
    } else {
      console.warn('missing source for dep', file);
    }
  } else {
    console.warn('got dep missing file or string id', file);
  }
  done();
}

function updateCacheOnTransformFile(b, moduleFile, dependentFile, done) {
  var cache = BrowserifyCache.getCache(b);
  if (cache.dependentFiles[dependentFile] == null) {
    cache.dependentFiles[dependentFile] = {};
  }
  cache.dependentFiles[dependentFile][moduleFile] = true;
  if (!cache.mtimes[dependentFile])
    return updateMtime(cache.mtimes, dependentFile, done);
  done();
}

function attachCachePersistHandler(b, cacheFile) {
  if (!cacheFile) return;

  b.on('bundle', function(bundleStream) {
    addCacheBlocker(b);
    bundleStream.on('end', function() {
      removeCacheBlocker(b);
    });
    // We need to wait until the cache is done being populated.
    // Use .once because the `b` browserify object can be re-used for multiple
    // bundles. We only want to save the cache once per bundle call.
    b.once('_cacheReadyToWrite', function() {
      storeCache(b, cacheFile);
    });
  });
}

function storeCache(b, cacheFile) {
  assertExists(cacheFile);

  var cache = BrowserifyCache.getCache(b);
  fs.writeFile(cacheFile, JSON.stringify(cache), {encoding: 'utf8'}, function(err) {
    if (err) b.emit('_cacheFileWriteError', err);
    else b.emit('_cacheFileWritten', cacheFile);
  });
}

function loadCacheData(b, cacheFile) {
  var cacheData = {};

  if (cacheFile) {
    try {
      cacheData = JSON.parse(fs.readFileSync(cacheFile, {encoding: 'utf8'}));
    } catch (err) {
      // no existing cache file
      b.emit('_cacheFileReadError', err);
    }
  }

  return cacheData;
}

function updateMtime(mtimes, file, done) {
  assertExists(mtimes);
  assertExists(file);

  fs.stat(file, function(err, stat) {
    if (!err) mtimes[file] = stat.mtime.getTime();
    done();
  });
}

module.exports = BrowserifyCache;
