(function() {
  // 可配置项
  var INDEX_URL = '/search.json'; // 搜索索引文件路径
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

    fetch(INDEX_URL)
      .then(function(r){ return r.json(); })
      .then(function(data){
        buildIndex(data);
      })
      .catch(function(err){
        console.error('加载搜索索引失败：', err);
        results.innerHTML = '<p class="text-danger">无法加载搜索索引</p>';
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
        results.innerHTML = '<p class="text-danger">搜索时发生错误</p>';
      }
    }, DEBOUNCE_MS);

    input.addEventListener('input', handler);
    input.addEventListener('keyup', function(e){
      if (e.key === 'Enter') input.blur();
    });
  }

  function buildIndex(data) {
    store = {};
    idx = lunr(function() {
      this.ref('url');
      this.field('title', { boost: 10 });
      this.field('content');

      data.forEach(function(doc) {
        this.add(doc);
        store[doc.url] = doc;
      }, this);
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatDate(dateString) {
    if (!dateString) return '';
    
    // 尝试解析日期字符串
    var date = new Date(dateString);
    if (isNaN(date.getTime())) return dateString;
    
    var options = { year: 'numeric', month: 'long', day: 'numeric' };
    return date.toLocaleDateString('en-US', options).replace(',', '');
  }

  function estimateReadTime(content) {
    if (!content) return '1 min read';
    
    // 估算阅读时间：按照60字/分钟的阅读速度
    var wordCount = content.split(/\s+/).length;
    var minutes = Math.max(1, Math.round(wordCount / 60));
    
    return minutes + ' min' + (minutes > 1 ? 's' : '') + ' read';
  }

  function renderResults(resultsArray, container) {
    if (!resultsArray || resultsArray.length === 0) {
      container.innerHTML = '<p class="text-muted">未找到匹配结果</p>';
      return;
    }
    
    var out = '';
    for (var i = 0; i < resultsArray.length; i++) {
      var r = resultsArray[i];
      var doc = store[r.ref];
      if (!doc) continue;
      
      out += '<div class="post-preview">';
      
      // 文章链接和标题
      out += '<a href="' + escapeHtml(doc.url) + '">';
      out += '<h2 class="post-title">' + escapeHtml(doc.title) + '</h2>';
      
      // 副标题 - 使用固定的"Plan and Realization"或从数据中获取
      var subtitle = doc.subtitle || 'Plan and Realization';
      out += '<h3 class="post-subtitle">' + escapeHtml(subtitle) + '</h3>';
      out += '</a>';
      
      // 元信息（作者、日期、阅读时间）
      out += '<p class="post-meta">Posted by Mrx on ' + formatDate(doc.date) + ' &middot; ' + estimateReadTime(doc.content) + '</p>';
      
      out += '</div>';
      
      // 添加分隔线（最后一个结果不添加）
      if (i < resultsArray.length - 1) {
        out += '<hr>';
      }
    }
    
    container.innerHTML = out;
  }

  // 页面加载完成后初始化
  document.addEventListener('DOMContentLoaded', init);
})();