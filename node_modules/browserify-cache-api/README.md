# browserify-cache-api

Attaches per-module caching of module resolution and transformation to a browserify instance.

Caches to a file on disk, invalidated by source file modification time.

Used by [browserify-incremental](https://github.com/jsdf/browserify-incremental)

```js
  // create a browserify instance
  var b = browserify({
    // cache and packageCache opts are required
    cache: {},
    packageCache: {},
    // and then your opts...
  });

  // attach caching, specifying a location to store the cache file
  browserifyCache(b, {cacheFile: './tmp/browserify-cache.json'});

  // browserify module resolution + transformation is now cached
```

## Contributing

Please see the [Contributor Guidelines](CONTRIBUTING.md).
