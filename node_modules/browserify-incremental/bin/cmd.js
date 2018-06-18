#!/usr/bin/env node

var browserifyIncremental = require('../');
var fs = require('fs');
var fromArgs = require('browserify/bin/args');
var JSONStream = require('JSONStream');
var through = require('through2');

function run() {
  process.stdout.on('error', process.exit);

  var b, outfile, verbose, cachefile;

  var b_ = fromArgs(process.argv.slice(2), browserifyIncremental.args);
  cachefile = b_.argv.cachefile || './browserify-cache.json';
  outfile = b_.argv.o || b_.argv.outfile;
  verbose = (b_.argv.v || b_.argv.verbose);
  b = browserifyIncremental(b_, {cacheFile: cachefile});

  b.on('update', function(changes) {
    if (verbose && changes.length) console.error('changed files:\n' + changes.join('\n'));
  });

  if (b.argv.version) {
    return console.log(require('../package.json').version);
  }

  b.on('error', errorExit);

  if (b.argv.pack) {
    process.stdin.pipe(b.pack()).pipe(process.stdout);
    process.stdin.resume();
    return;
  }

  if (b.argv.deps) {
    var stringify = JSONStream.stringify();
    stringify.pipe(process.stdout);
    b.pipeline.get('deps').push(through.obj(
      function(row, enc, next) { stringify.write(row); next(); },
      function() { stringify.end(); }
    ));
    return b.bundle();
  }

  if (b.argv.list) {
    b.pipeline.get('deps').push(through.obj(function(row, enc, next) {
      console.log(row.file || row.id);
      next();
    }));
    return b.bundle();
  }

  var bytes, time;
  b.on('bytes', function(b) { bytes = b; });
  b.on('time', function(t) { time = t; });

  var bundle = b.bundle();
  bundle.on('error', errorExit);

  bundle.on('end', function() {
    if (verbose) {
      console.error(bytes + ' bytes written to ' + (outfile || 'stdout')
        + ' (' + (time / 1000).toFixed(2) + ' seconds)'
      );
    }
  });

  if (outfile) {
    bundle.pipe(fs.createWriteStream(outfile));
  }
  else {
    bundle.pipe(process.stdout);
  }
}

function errorExit(err) {
  if (err.stack) {
    console.error(err.stack);
  }
  else {
    console.error(String(err));
  }
  process.exit(1);
}

run();
