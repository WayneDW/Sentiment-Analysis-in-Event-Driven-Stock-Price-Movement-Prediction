function template(locals) {
    var pug_html = "", pug_mixins = {}, pug_interp;
    pug_html = pug_html + '<div :my-var="model"></div><span v-for="item in items" :key="item.id" :value="item.name"></span><span v-for="item in items" :key="item.id" :value="item.name"></span><a :link="goHere" value="static" :my-value="dynamic" @click="onClick()" :another="more">Click Me!</a>';
    return pug_html;
}