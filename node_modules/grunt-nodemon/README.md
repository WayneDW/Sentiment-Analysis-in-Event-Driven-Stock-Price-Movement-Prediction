# Note: This is not actively maintained, please make an issue if you are interested in helping maintain this project.

# grunt-nodemon

> Run [nodemon](https://github.com/remy/nodemon) as a grunt task for easy configuration and integration with the rest of your workflow

[![NPM version](https://badge.fury.io/js/grunt-nodemon.png)](http://badge.fury.io/js/grunt-nodemon) [![Dependency Status](https://david-dm.org/ChrisWren/grunt-nodemon.png)](https://david-dm.org/ChrisWren/grunt-nodemon) [![Travis Status](https://travis-ci.org/ChrisWren/grunt-nodemon.png)](https://travis-ci.org/ChrisWren/grunt-nodemon)

## Getting Started
If you haven't used grunt before, be sure to check out the [Getting Started](http://gruntjs.com/getting-started) guide, as it explains how to create a gruntfile as well as install and use grunt plugins. Once you're familiar with that process, install this plugin with this command:
```shell
npm install grunt-nodemon --save-dev
```

Then add this line to your project's `Gruntfile.js` gruntfile:

```javascript
grunt.loadNpmTasks('grunt-nodemon');
```

## Documentation

### Minimal Usage
The minimal usage of grunt-nodemon runs with a `script` specified:

```js
nodemon: {
  dev: {
    script: 'index.js'
  }
}
```

### Usage with all available options set

```js
nodemon: {
  dev: {
    script: 'index.js',
    options: {
      args: ['dev'],
      nodeArgs: ['--debug'],
      callback: function (nodemon) {
        nodemon.on('log', function (event) {
          console.log(event.colour);
        });
      },
      env: {
        PORT: '8181'
      },
      cwd: __dirname,
      ignore: ['node_modules/**'],
      ext: 'js,coffee',
      watch: ['server'],
      delay: 1000,
      legacyWatch: true
    }
  },
  exec: {
    options: {
      exec: 'less'
    }
  }
}
```

### Advanced Usage

A common use case is to run `nodemon` with other tasks concurrently. It is also common to open a browser tab when starting a server, and reload that tab when the server code changes. These workflows can be achieved with the following config, which uses a custom [`options.callback`](#callback) function, and [grunt-concurrent](https://github.com/sindresorhus/grunt-concurrent) to run nodemon, [node-inspector](https://github.com/ChrisWren/grunt-node-inspector), and [watch](https://github.com/gruntjs/grunt-contrib-watch) in a single terminal tab:

```js
concurrent: {
  dev: {
    tasks: ['nodemon', 'node-inspector', 'watch'],
    options: {
      logConcurrentOutput: true
    }
  }
},
nodemon: {
  dev: {
    script: 'index.js',
    options: {
      nodeArgs: ['--debug'],
      env: {
        PORT: '5455'
      },
      // omit this property if you aren't serving HTML files and 
      // don't want to open a browser tab on start
      callback: function (nodemon) {
        nodemon.on('log', function (event) {
          console.log(event.colour);
        });
        
        // opens browser on initial server start
        nodemon.on('config:update', function () {
          // Delay before server listens on port
          setTimeout(function() {
            require('open')('http://localhost:5455');
          }, 1000);
        });

        // refreshes browser when server reboots
        nodemon.on('restart', function () {
          // Delay before server listens on port
          setTimeout(function() {
            require('fs').writeFileSync('.rebooted', 'rebooted');
          }, 1000);
        });
      }
    }
  }
},
watch: {
  server: {
    files: ['.rebooted'],
    options: {
      livereload: true
    }
  } 
}
```

*Note that using the callback config above assumes you have `open` installed and are injecting a LiveReload script into your HTML file(s). You can use [grunt-inject](https://github.com/ChrisWren/grunt-inject) to inject the LiveReload script.*

### Required property

#### script
Type: `String`

Script that nodemon runs and restarts when changes are detected.

### Options

#### args
Type: `Array` of `Strings`

List of arguments to be passed to your script.

#### nodeArgs
Type: `Array` of `Strings`

List of arguments to be passed to node. The most common argument is `--debug` or `--debug-brk` to start a debugging server.

#### callback
Type:  `Function`
Default:

```js
function(nodemon) {
  // By default the nodemon output is logged
  nodemon.on('log', function(event) {
    console.log(event.colour);
  });
};
```

Callback which receives the `nodemon` object. This can be used to respond to changes in a running app, and then do cool things like LiveReload a web browser when the app restarts. See the [nodemon docs](https://github.com/remy/nodemon/blob/master/doc/events.md#states) for the full list of events you can tap into.

#### ignore
Type: `Array` of `String globs` Default: `['node_modules/**']`

List of ignored files specified by a glob pattern relative to the [watch](#watch)ed folder. [Here](https://github.com/remy/nodemon#ignoring-files) is an explanation of how to use the patterns to ignore files.

#### ext
Type: `String` Default: `'js'`

String with comma separated file extensions to watch. By default, nodemon watches `.js` files.

#### watch
Type: `Array` of `Strings` Default: `['.']`

List of folders to watch for changes. By default nodemon will traverse sub-directories, so there's no need in explicitly including sub-directories.

#### delay
Type: `Number` Default: `1000`

Delay the restart of nodemon by a number of milliseconds when compiling a large amount of files so that the app doesn't needlessly restart after each file is changed.

#### legacyWatch
Type: `Boolean` Default: `false`

If you wish to force nodemon to start with the legacy watch method. See <https://github.com/remy/nodemon/blob/master/faq.md#help-my-changes-arent-being-detected> for more details.

#### cwd
Type: `String`

The current working directory to run your script from.

#### env
Type: `Object`

Hash of environment variables to pass to your script.

#### exec
Type: `String`

You can use nodemon to execute a command outside of node. Use this option to specify a command as a string with the argument being the script parameter above. You can read more on exec [here](https://github.com/remy/nodemon#running-non-node-scripts).

# Changelog

**0.3.0** - Updated to nodemon `1.2.0`.

**0.2.1** - Updated README on npmjs.org with correct options.

**0.2.0** - Updated to nodemon 1.0, added new [`callback`](#callback) option.

**Breaking changes:**

- `options.file` is now `script` and is a required property. Some properties were changed to match nodemon: `ignoredFiles` -> `ignore`, `watchedFolders` -> `watch`, `delayTime` -> `delay`, `watchedExtensions` -> `ext`(now a string) to match nodemon.

**0.1.2** - `nodemon` can now be listed as a dependency in the package.json and grunt-nodemon will resolve the nodemon.js file's location correctly.

**0.1.1** - Added `legacyWatch` option thanks to [@jonursenbach](https://github.com/jonursenbach).

**0.1.0** - Removed `debug` and `debugBrk` options as they are encapsulated by the `nodeArgs` option.

**Breaking changes:**

- Configs with the `debug` or `debugBrk` options will no longer work as expected. They simply need to be added to `nodeArgs`.

**0.0.10** - Added `nodeArgs` option thanks to [@eugeneiiim](https://github.com/eugeneiiim).

**0.0.9** - Fixed bug when using `cwd` with `ignoredFiles`.

**0.0.8** - Added error logging for incorrectly installed `nodemon`.

**0.0.7** - Added `debugBreak` option thanks to [@bchu](https://github.com/bchu).

**0.0.6** - Added `env` option.

**0.0.5** - Added `cwd` option.

**0.0.4** - Added `nodemon` as a proper dependency.

**0.0.3** - Uses local version of `nodemon` for convenience and versioning.

**0.0.2** - Removes `.nodemonignore` if it was previously generated and then the `ignoredFiles` option is removed.

**0.0.1** - Added warning if `nodemon` isn't installed as a global module.

**0.0.0** - Initial release
