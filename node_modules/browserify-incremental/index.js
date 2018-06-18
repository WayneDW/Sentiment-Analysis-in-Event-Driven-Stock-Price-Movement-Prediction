var xtend = require('xtend');
var through = require('through2');
var browserify = require('browserify');
var browserifyCache = require('browserify-cache-api');

module.exports = browserifyIncremental;
browserifyIncremental.browserify = browserify;
browserifyIncremental.args = browserifyCache.args;

function browserifyIncremental(files, opts) {
  var b; // browserify instance

  // browserify plugin boilerplate, normalises variable arguments
  if (files && typeof files.bundle === 'function') {
    // browserify instance as first arg
    b = files;
    opts = opts || b._options;
  } else {
    if (!opts) {
      // opts as first arg (or no args)
      opts = files || {};
      files = undefined;
      b = browserify(xtend(browserifyCache.args, opts));
    } else {
      // files as first arg, opts as second arg
      b = browserify(files, xtend(browserifyCache.args, opts));
    }
  }

  browserifyCache(b, opts);

  if (!b.pipeline) b.emit('error', new Error('missing pipeline: incompatible browserify version (< 5.x)'));
  b.on('reset', function() { attachMetrics(b); });
  attachMetrics(b);

  return b;
}

function attachMetrics(b) {
  var time = null;
  var bytes = 0;
  b.pipeline.get('record').on('end', function() {
    time = Date.now();
  });

  b.pipeline.get('wrap').push(through(write, end));
  function write(buf, enc, next) {
    bytes += buf.length;
    this.push(buf);
    next();
  }
  function end() {
    var delta = Date.now() - time;
    b.emit('time', delta);
    b.emit('bytes', bytes);
    b.emit('log', bytes + ' bytes written ('
        + (delta / 1000).toFixed(2) + ' seconds)'
    );
    this.push(null);
  }
}
