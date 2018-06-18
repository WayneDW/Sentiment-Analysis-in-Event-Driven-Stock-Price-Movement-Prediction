/*
 * grunt-nodemon
 * https://github.com/ChrisWren/grunt-nodemon
 *
 * Copyright (c) 2014 Chris Wren and contributors
 * Licensed under the MIT license.
 */
var nodemon = require('nodemon');

module.exports = function (grunt) {
  'use strict';

  grunt.registerMultiTask('nodemon', 'Runs a nodemon monitor of your node.js server.', function () {

    this.async();
    var options = this.options();

    options.script = this.data.script;

    var callback;

    if (options.callback) {
      callback = options.callback;
      delete options.callback;
    } else {
      callback = function(nodemonApp) {
        nodemonApp.on('log', function (event) {
          console.log(event.colour);
        });
      };
    }

    callback(nodemon(options));
  });
};
