function proxyEvent(source, target, name) {
  source.on(name, function() {
    target.emit.apply(target, [name].concat([].slice.call(arguments)));
  });
}

module.exports = proxyEvent;
