// assets/vendor/startbootstrap-clean-blog/js/search.js
(function() {
  // configurable:
  var INDEX_URL = '/search.json'; // 如果你把 search.json 放在别的位置，这里改为相对路径
  var INPUT_ID = 'search-input';
  var RESULTS_ID = 'search-results';
  var DEBOUNCE_MS = 200;

  var idx = null;
  var store = {};

  function debounce(fn, wait) {
    var t;
    return function() {
      var args = arguments;
      clearTimeout(t);
      t = setTimeout(function(){ fn.apply(null, args); }, wait);
    };
  }

  function init() {
    var input = document.getElementById(INPUT_ID);
    var results = document.getElementById(RESULTS_ID);
    if (!input || !results) return;

    results.innerHTML = '<p>正在加载索引…</p>';

    fetch(INDEX_URL)
      .then(function(r){ return r.json(); })
      .then(function(data){
        buildIndex(data);
        results.innerHTML = '<p>索引加载完成，输入关键词开始搜索。</p>';
      })
      .catch(function(err){
        console.error('加载搜索索引失败：', err);
        results.innerHTML = '<p class="text-danger">加载搜索索引失败，请检查 /search.json 是否存在。</p>';
      });

    var handler = debounce(function(e){
      var q = e.target.value.trim();
      if (!q) {
        results.innerHTML = '';
        return;
      }
      if (!idx) {
        results.innerHTML = '<p>索引尚未就绪，请稍候…</p>';
        return;
      }
      try {
        var lunrResults = idx.search(q);
        renderResults(lunrResults, results);
      } catch (err) {
        console.error('搜索出错：', err);
        results.innerHTML = '<p class="text-danger">搜索时发生错误（请尝试更简单的关键词）。</p>';
      }
    }, DEBOUNCE_MS);

    input.addEventListener('input', handler);
    input.addEventListener('keyup', function(e){
      if (e.key === 'Enter') input.blur();
    });
  }

  function buildIndex(data) {
    store = {};
    idx = lunr(function () {
      this.use(lunr.zh); // 启用中文支持
      this.ref('url');
      this.field('title', { boost: 10 });
      this.field('content');
      for (var i = 0; i < data.length; i++) {
        this.add(data[i]);
        store[data[i].url] = data[i];
      }
    });
  }

  function snippet(text, q, len) {
    len = len || 200;
    if (!text) return '';
    if (text.length <= len) return text;
    // 简单截取前段作为摘要
    return text.slice(0, len).trim() + '…';
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function renderResults(resultsArray, container) {
    if (!resultsArray || resultsArray.length === 0) {
      container.innerHTML = '<p>未找到结果。</p>';
      return;
    }
    var out = '<div class="list-group">';
    for (var i=0; i<resultsArray.length; i++) {
      var r = resultsArray[i];
      var doc = store[r.ref];
      if (!doc) continue;
      out += '<a class="list-group-item list-group-item-action mb-2" href="' + escapeHtml(doc.url) + '">';
      out += '<h5 class="mb-1">' + escapeHtml(doc.title) + '</h5>';
      out += '<small class="text-muted">' + (doc.date ? escapeHtml(doc.date) : '') + '</small>';
      out += '<p class="mb-1 mt-2">' + escapeHtml(snippet(doc.content, '', 200)) + '</p>';
      out += '</a>';
    }
    out += '</div>';
    container.innerHTML = out;
  }

  // DOMContentLoaded 保证在页面加载完成后运行
  document.addEventListener('DOMContentLoaded', init);
})();
