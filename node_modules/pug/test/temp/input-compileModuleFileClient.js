var pug = require("pug-runtime");function template(locals) {var pug_html = "", pug_mixins = {}, pug_interp;var pug_debug_filename, pug_debug_line;try {var self = locals || {};;pug_debug_line = 1;
pug_html = pug_html + "\u003Cdiv class=\"bar\"\u003E";
;pug_debug_line = 1;
pug_html = pug_html + (pug.escape(null == (pug_interp = self.foo) ? "" : pug_interp)) + "\u003C\u002Fdiv\u003E";} catch (err) {pug.rethrow(err, pug_debug_filename, pug_debug_line);};return pug_html;} module.exports = template;