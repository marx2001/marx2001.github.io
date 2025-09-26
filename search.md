---
layout: page
title: Search
permalink: /search/
---

<div class="search-container">
  <div class="search-box">
    <input type="text" id="search-input" placeholder="Search posts..." class="form-control">
    <button id="search-button" class="btn btn-primary">
      <i class="fas fa-search"></i> Search
    </button>
  </div>
  
  <div id="search-results" class="mt-4">
    <div class="initial-message">
      <p>Enter keywords to search blog posts.</p>
    </div>
  </div>
</div>

<!-- 使用正确的路径引用 search.js -->
<script src="{{ '/assets/vendor/startbootstrap-clean-blog/js/search.js' | relative_url }}"></script>