var _ = require('lodash');
var path = require('path');
var resolve = require('resolve');
var glob = require('glob');
var browserifyInc = require('browserify-incremental');

module.exports = GruntBrowserifyRunner;

function GruntBrowserifyRunner(options) {
  this.browserify = options.browserify;
  this.watchify = options.watchify;
  this.logger = options.logger;
  this.writer = options.writer;
  this.firstBuild = true;
}

// persist the created browserify instances between calls from grunt-contrib-watch
// this is specifically for `options.cacheFile`, `browserify-incremental`
var destinations = {}

GruntBrowserifyRunner.prototype = _.create(GruntBrowserifyRunner.prototype, {
  run: function (files, destination, options, next) {
    var self = this;

    //set constructor options and instantiate
    var bOpts = _.cloneDeep(options.browserifyOptions) || {};
    bOpts.entries = bOpts.entries || files;

    // watchify options
    var wOpts = options.watchifyOptions || {};

    //determine watchify or browserify-incremental or browserify
    var b
    if (options.watch) {
      bOpts = _.extend({ cache: {}, packageCache: {} }, bOpts);
      b = this.watchify(this.browserify(bOpts), wOpts)
    } else if (options.cacheFile) {
      if (destinations[destination]) {
        return destinations[destination].call(this, next)
      }
      bOpts = Object.assign({}, bOpts, browserifyInc.args)
      b = this.browserify(bOpts)
      browserifyInc(b, {cacheFile: options.cacheFile})

      b.on('log', function (msg) {
        self.logger.log.ok(msg.cyan)
      })

      destinations[destination] = function (done) {
        doBundle(b, options, this.onBundleComplete(destination, options, done))
      }
    } else {
      b = this.browserify(bOpts)
    }

    b.on('error', function (err) {
      self.logger.fail.warn(err);
    });

    if(options.bundleOptions) {
      throw new Error('bundleOptions is no longer used. Move all option in browserifyOptions.');
    }

    if(options.alias) {
      if(_.isPlainObject(options.alias)) {
        for(var alias in options.alias) {
          b.require(options.alias[alias], {expose: alias});
        }
      }
      else {
        requireFiles(b, options.alias);
      }
    }

    if(options.require) {
      requireFiles(b, options.require);
    }

    if (options.exclude) {
      _.forEach(options.exclude, function (file) {
        runOptionForGlob(b, 'exclude', file);
      });
    }

    if (options.ignore) {
      _.forEach(options.ignore, function (file) {
        runOptionForGlob(b, 'ignore', file);
      });
    }

    if (options.external) {
      // allow externalizing of alias object
      if(_.isPlainObject(options.external)) {
        for(var id in options.external) {
          if (testForGlob(id)) {
            runOptionForGlob(b, 'external', id);
          }
          else {
            b.external(id);
          }
        }
      }
      else {
        _.forEach(options.external, function (id) {
          //allow externalizing of require lists
          if (id.match(':')) {
            id = id.split(':')[1];
          }

          if (testForGlob(id)) {
            runOptionForGlob(b, 'external', id);
          }
          else {
            b.external(id);
          }
        });
      }
    }

    if (options.transform) {
      _.forEach(options.transform, function (transformer) {
        b.transform(transformer);
      });
    }

    if (options.plugin) {
      _.forEach(options.plugin, function (plugin) {
        b.plugin(plugin);
      });
    }


    var destPath = this.createDestDir(destination);
    var keepAlive = this.keepAliveFn.bind(this, destination);
    var done = options.keepAlive? keepAlive : next;
    var bundleComplete = this.onBundleComplete(destination, options, done);

    if (options.watch) {
      var bundleUpdate = this.onBundleComplete(destination, options, keepAlive);
      b.on('update', function (ids) {
        ids.forEach(function (id) {
          self.logger.log.ok(id.cyan + ' changed, updating bundle.');
        });
        doBundle(b, options, bundleUpdate);
      });
    }

    if (options.configure) {
      options.configure(b);
    }

    doBundle(b, options, bundleComplete);
  },

  createDestDir: function (destination) {
    var destPath = path.dirname(path.resolve(destination));
    if (!this.writer.exists(destPath)) {
      this.writer.mkdir(destPath);
    }
    return destPath;
  },

  keepAliveFn: function (destination) {
    //this.logger.log.ok('Watchifying...');
  },

  onBundleComplete: function (destination, options, next) {
    var self = this;
    return function (err, buf) {
      if (err) {
        self.logger.log.error(err);
        if (self.firstBuild || !options.keepAlive) {
          self.logger.fail.warn('Error running grunt-browserify.');
        }
      }
      else if (buf) {
          // prepend the banner
          if(options.banner) {
              buf = Buffer.concat([new Buffer(options.banner + '\n', 'utf8'), buf]);
          }

        self.logger.log.ok('Bundle ' + destination.cyan + ' created. ' + (options.keepAlive ? 'Watchifying...' : ''));
        self.writer.write(destination, buf);
      }

      self.firstBuild = false;
      next();
    };
  }
});

function doBundle(browserifyInstance, opts, bundleComplete) {
  if (opts.preBundleCB) {
    opts.preBundleCB(browserifyInstance);
  }

  browserifyInstance.bundle(function (err, buf) {
    if (opts.postBundleCB) {
      opts.postBundleCB(err, buf, bundleComplete);
    }
    else {
      bundleComplete(err, buf);
    }
  })
  // Need this to ensure it works on MacOS
  // See 'important' note here: https://github.com/substack/watchify#var-w--watchifyb-opts
  .on('data', function() {});
}

function testForGlob(id) {
  return (/\*/.test(id));
}

function runOptionForGlob(browserifyInstance, method, pattern) {
  var files = glob.sync(pattern);
  if (!files || files.length < 1) {
    //it's not a glob, it's a file / module path
    files = [pattern];
  }
  files.forEach(function (f) {
    browserifyInstance[method].call(browserifyInstance, f);
  });
}

function requireFiles(b, requiredFiles) {
  _.forEach(requiredFiles, function (file) {
    var filePath, opts;
    if (Array.isArray(file)) {
      filePath = file[0];
      opts = file[1];
    }
    else {
      var filePair = file.split(':');
      filePath = filePair[0];
      opts = {
        expose: filePair.length === 1 ? filePair[0] : filePair[1]
      };
    }
    b.require(filePath, opts);
  });
}
