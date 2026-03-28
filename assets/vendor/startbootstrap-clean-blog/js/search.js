(function () {
  var INDEX_URL = '/search.json';
  var INPUT_ID = 'search-input';
  var RESULTS_ID = 'search-results';
  var DEBOUNCE_MS = 250;
  var MAX_RESULTS = 50;

  var idx = null;
  var store = {};
  var docs = [];
  var indexReady = false;
  var loadingIndex = false;
  var pendingQuery = '';

  function debounce(fn, wait) {
    var t;
    return function () {
      var args = arguments;
      clearTimeout(t);
      t = setTimeout(function () {
        fn.apply(null, args);
      }, wait);
    };
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function normalizeText(str) {
    return String(str || '')
      .toLowerCase()
      .replace(/\u3000/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function uniqueArray(arr) {
    var seen = {};
    var out = [];
    for (var i = 0; i < arr.length; i++) {
      var v = arr[i];
      if (!v) continue;
      if (!seen[v]) {
        seen[v] = true;
        out.push(v);
      }
    }
    return out;
  }

  function escapeLunrTerm(term) {
    return String(term || '').replace(/[+\-!(){}\[\]^"~*?:\\\/]/g, ' ');
  }

  function tokenizeQuery(q) {
    var raw = normalizeText(q);
    if (!raw) return [];

    var tokens = [raw];

    var parts = raw.split(/\s+/);
    for (var i = 0; i < parts.length; i++) {
      if (parts[i]) tokens.push(parts[i]);
    }

    var extra = raw.match(/[a-z0-9_-]+/g) || [];
    for (var j = 0; j < extra.length; j++) {
      tokens.push(extra[j]);
    }

    var chineseOnly = raw.replace(/[^\u4e00-\u9fff]/g, '');
    if (chineseOnly.length >= 2) {
      for (var n = 2; n <= 3; n++) {
        for (var k = 0; k <= chineseOnly.length - n; k++) {
          tokens.push(chineseOnly.slice(k, k + n));
        }
      }
    }

    tokens = tokens
      .map(function (t) {
        return normalizeText(escapeLunrTerm(t));
      })
      .filter(function (t) {
        return !!t;
      });

    return uniqueArray(tokens);
  }

  function formatDate(dateString) {
    if (!dateString) return '';
    var date = new Date(dateString);
    if (isNaN(date.getTime())) return dateString;
    var options = { year: 'numeric', month: 'long', day: 'numeric' };
    return date.toLocaleDateString('en-US', options).replace(',', '');
  }

  function estimateReadTime(content) {
    if (!content) return '1 min read';
    var text = String(content).replace(/\s+/g, '');
    var minutes = Math.max(1, Math.round(text.length / 300));
    return minutes + ' min' + (minutes > 1 ? 's' : '') + ' read';
  }

  function buildSnippet(doc, query) {
    var text = String(doc.content || '');
    if (!text) return '';

    var normalizedText = normalizeText(text);
    var normalizedQuery = normalizeText(query);
    var pos = normalizedText.indexOf(normalizedQuery);

    if (pos === -1) {
      var tokens = tokenizeQuery(query);
      for (var i = 0; i < tokens.length; i++) {
        pos = normalizedText.indexOf(tokens[i]);
        if (pos !== -1) break;
      }
    }

    if (pos === -1) {
      return text.slice(0, 120) + (text.length > 120 ? '...' : '');
    }

    var start = Math.max(0, pos - 30);
    var end = Math.min(text.length, pos + 100);
    var snippet = text.slice(start, end);

    if (start > 0) snippet = '...' + snippet;
    if (end < text.length) snippet += '...';

    return snippet;
  }

  function highlightText(text, query) {
    var safe = escapeHtml(text || '');
    var tokens = tokenizeQuery(query).sort(function (a, b) {
      return b.length - a.length;
    });

    for (var i = 0; i < tokens.length; i++) {
      var t = tokens[i];
      if (!t || t.length < 2) continue;
      var escaped = t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      safe = safe.replace(new RegExp('(' + escaped + ')', 'gi'), '<mark>$1</mark>');
    }

    return safe;
  }

  function loadIndex(callback) {
    if (indexReady) {
      callback();
      return;
    }

    if (loadingIndex) return;

    loadingIndex = true;

    var results = document.getElementById(RESULTS_ID);
    if (results) {
      results.innerHTML = '<p class="text-muted">正在加载搜索索引...</p>';
    }

    fetch(INDEX_URL)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        buildIndex(data || []);
        indexReady = true;
        loadingIndex = false;
        callback();
      })
      .catch(function (err) {
        loadingIndex = false;
        console.error('加载搜索索引失败：', err);
        if (results) {
          results.innerHTML = '<p class="text-danger">无法加载搜索索引</p>';
        }
      });
  }

  function buildIndex(data) {
    store = {};
    docs = [];

    data.forEach(function (doc) {
      var safeDoc = {
        title: doc.title || '',
        url: doc.url || '',
        date: doc.date || '',
        content: doc.content || '',
        text: (doc.title || '') + ' ' + (doc.content || '')
      };
      store[safeDoc.url] = safeDoc;
      docs.push(safeDoc);
    });

    idx = lunr(function () {
      this.ref('url');
      this.field('title', { boost: 20 });
      this.field('content', { boost: 8 });
      this.field('text', { boost: 5 });

      docs.forEach(function (doc) {
        this.add(doc);
      }, this);
    });
  }

  function smartSearch(query) {
    var tokens = tokenizeQuery(query);
    var resultMap = {};

    function addResult(ref, score) {
      if (!ref || !store[ref]) return;
      if (!resultMap[ref]) {
        resultMap[ref] = { ref: ref, score: score || 0 };
      } else {
        resultMap[ref].score = Math.max(resultMap[ref].score, score || 0);
      }
    }

    try {
      var direct = idx.search(query);
      for (var i = 0; i < direct.length; i++) {
        addResult(direct[i].ref, 1000 + direct[i].score);
      }
    } catch (e) {}

    try {
      var flexible = idx.query(function (q) {
        for (var i = 0; i < tokens.length; i++) {
          var term = tokens[i];
          if (!term) continue;

          q.term(term, {
            fields: ['title'],
            boost: 30,
            presence: lunr.Query.presence.OPTIONAL
          });

          q.term(term, {
            fields: ['content'],
            boost: 10,
            presence: lunr.Query.presence.OPTIONAL
          });

          if (term.length >= 2) {
            q.term(term, {
              fields: ['title'],
              boost: 15,
              wildcard: lunr.Query.wildcard.TRAILING,
              presence: lunr.Query.presence.OPTIONAL
            });

            q.term(term, {
              fields: ['content'],
              boost: 5,
              wildcard: lunr.Query.wildcard.TRAILING,
              presence: lunr.Query.presence.OPTIONAL
            });
          }
        }
      });

      for (var j = 0; j < flexible.length; j++) {
        addResult(flexible[j].ref, 500 + flexible[j].score);
      }
    } catch (e2) {}

    var nq = normalizeText(query);
    for (var k = 0; k < docs.length; k++) {
      var doc = docs[k];
      var title = normalizeText(doc.title);
      var content = normalizeText(doc.content);

      var score = 0;

      if (title.indexOf(nq) !== -1) score += 300;
      if (content.indexOf(nq) !== -1) score += 180;

      for (var t = 0; t < tokens.length; t++) {
        var token = tokens[t];
        if (!token) continue;

        if (title.indexOf(token) !== -1) score += 40;
        if (content.indexOf(token) !== -1) score += 15;
      }

      if (score > 0) addResult(doc.url, score);
    }

    var merged = Object.keys(resultMap).map(function (ref) {
      return resultMap[ref];
    });

    merged.sort(function (a, b) {
      var docA = store[a.ref] || {};
      var docB = store[b.ref] || {};
      var qn = normalizeText(query);

      var aTitleHit = normalizeText(docA.title).indexOf(qn) !== -1 ? 1 : 0;
      var bTitleHit = normalizeText(docB.title).indexOf(qn) !== -1 ? 1 : 0;

      if (aTitleHit !== bTitleHit) return bTitleHit - aTitleHit;
      return b.score - a.score;
    });

    return merged.slice(0, MAX_RESULTS);
  }

  function renderResults(resultsArray, container, query) {
    if (!resultsArray || resultsArray.length === 0) {
      container.innerHTML = '<p class="text-muted">未找到匹配结果</p>';
      return;
    }

    var out = '<p class="text-muted mb-4">共找到 ' + resultsArray.length + ' 篇相关文章</p>';

    for (var i = 0; i < resultsArray.length; i++) {
      var r = resultsArray[i];
      var doc = store[r.ref];
      if (!doc) continue;

      var snippet = buildSnippet(doc, query);

      out += '<div class="post-preview">';
      out += '<a href="' + escapeHtml(doc.url) + '">';
      out += '<h2 class="post-title">' + highlightText(doc.title, query) + '</h2>';
      out += '</a>';

      out += '<p class="post-meta">Posted by Mrx on ' +
        escapeHtml(formatDate(doc.date)) +
        ' &middot; ' +
        escapeHtml(estimateReadTime(doc.content)) +
        '</p>';

      if (snippet) {
        out += '<p class="post-entry">' + highlightText(snippet, query) + '</p>';
      }

      out += '</div>';

      if (i < resultsArray.length - 1) {
        out += '<hr>';
      }
    }

    container.innerHTML = out;
  }

  function init() {
    var input = document.getElementById(INPUT_ID);
    var results = document.getElementById(RESULTS_ID);
    if (!input || !results) return;

    var handler = debounce(function (e) {
      var q = (e.target.value || '').trim();

      if (!q) {
        pendingQuery = '';
        results.innerHTML = '';
        return;
      }

      pendingQuery = q;

      if (!indexReady) {
        loadIndex(function () {
          if (!pendingQuery) return;
          var mergedResults = smartSearch(pendingQuery);
          renderResults(mergedResults, results, pendingQuery);
        });
        return;
      }

      var mergedResults = smartSearch(q);
      renderResults(mergedResults, results, q);
    }, DEBOUNCE_MS);

    input.addEventListener('input', handler);
    input.addEventListener('keyup', function (e) {
      if (e.key === 'Enter') input.blur();
    });
  }

  document.addEventListener('DOMContentLoaded', init);
})();