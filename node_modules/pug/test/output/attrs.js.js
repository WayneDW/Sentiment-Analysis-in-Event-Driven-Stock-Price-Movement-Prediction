function pug_attr(t, e, n, f) {
    return !1 !== e && null != e && (e || "class" !== t && "style" !== t) ? !0 === e ? " " + (f ? t : t + '="' + t + '"') : ("function" == typeof e.toJSON && (e = e.toJSON()), 
    "string" == typeof e || (e = JSON.stringify(e), n || -1 === e.indexOf('"')) ? (n && (e = pug_escape(e)), 
    " " + t + '="' + e + '"') : " " + t + "='" + e.replace(/'/g, "&#39;") + "'") : "";
}

function pug_attrs(t, r) {
    var a = "";
    for (var s in t) if (pug_has_own_property.call(t, s)) {
        var u = t[s];
        if ("class" === s) {
            u = pug_classes(u), a = pug_attr(s, u, !1, r) + a;
            continue;
        }
        "style" === s && (u = pug_style(u)), a += pug_attr(s, u, !1, r);
    }
    return a;
}

function pug_classes(s, r) {
    return Array.isArray(s) ? pug_classes_array(s, r) : s && "object" == typeof s ? pug_classes_object(s) : s || "";
}

function pug_classes_array(r, a) {
    for (var s, e = "", u = "", c = Array.isArray(a), g = 0; g < r.length; g++) (s = pug_classes(r[g])) && (c && a[g] && (s = pug_escape(s)), 
    e = e + u + s, u = " ");
    return e;
}

function pug_classes_object(r) {
    var a = "", n = "";
    for (var o in r) o && r[o] && pug_has_own_property.call(r, o) && (a = a + n + o, 
    n = " ");
    return a;
}

function pug_escape(e) {
    var a = "" + e, t = pug_match_html.exec(a);
    if (!t) return e;
    var r, c, n, s = "";
    for (r = t.index, c = 0; r < a.length; r++) {
        switch (a.charCodeAt(r)) {
          case 34:
            n = "&quot;";
            break;

          case 38:
            n = "&amp;";
            break;

          case 60:
            n = "&lt;";
            break;

          case 62:
            n = "&gt;";
            break;

          default:
            continue;
        }
        c !== r && (s += a.substring(c, r)), c = r + 1, s += n;
    }
    return c !== r ? s + a.substring(c, r) : s;
}

var pug_has_own_property = Object.prototype.hasOwnProperty;

var pug_match_html = /["&<>]/;

function pug_merge(e, r) {
    if (1 === arguments.length) {
        for (var t = e[0], g = 1; g < e.length; g++) t = pug_merge(t, e[g]);
        return t;
    }
    for (var l in r) if ("class" === l) {
        var n = e[l] || [];
        e[l] = (Array.isArray(n) ? n : [ n ]).concat(r[l] || []);
    } else if ("style" === l) {
        var n = pug_style(e[l]);
        n = n && ";" !== n[n.length - 1] ? n + ";" : n;
        var a = pug_style(r[l]);
        a = a && ";" !== a[a.length - 1] ? a + ";" : a, e[l] = n + a;
    } else e[l] = r[l];
    return e;
}

function pug_style(r) {
    if (!r) return "";
    if ("object" == typeof r) {
        var t = "";
        for (var e in r) pug_has_own_property.call(r, e) && (t = t + e + ":" + r[e] + ";");
        return t;
    }
    return r + "";
}

function template(locals) {
    var pug_html = "", pug_mixins = {}, pug_interp;
    var id = 5;
    function answer() {
        return 42;
    }
    pug_html = pug_html + "<a" + (' class="button"' + pug_attr("href", "/user/" + id, true, false)) + "></a><a" + (' class="button"' + pug_attr("href", "/user/" + id, true, false)) + "></a><meta" + (' key="answer"' + pug_attr("value", answer(), true, false)) + '/><a class="class1 class2"></a><a class="tag-class class1 class2"></a><a' + (' class="button"' + pug_attr("href", "/user/" + id, true, false)) + "></a><a" + (' class="button"' + pug_attr("href", "/user/" + id, true, false)) + "></a><meta" + (' key="answer"' + pug_attr("value", answer(), true, false)) + '/><a class="class1 class2"></a><a class="tag-class class1 class2"></a><div' + pug_attrs(pug_merge([ {
        id: pug_escape(id)
    }, {
        foo: "bar"
    } ]), false) + "></div>";
    var bar = null;
    pug_html = pug_html + "<div" + pug_attrs(pug_merge([ {
        foo: null,
        bar: pug_escape(bar)
    }, {
        baz: "baz"
    } ]), false) + "></div>";
    return pug_html;
}