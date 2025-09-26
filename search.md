---
layout: page
title: Search
permalink: /search/
---

<div class="search-container">
  <div class="search-box">
    <input type="text" id="search-input" placeholder="Search posts..." class="form-control">
    <button id="search-button" class="btn btn-primary">
      <i class="fas fa-search"></i>
    </button>
  </div>
  
  <div id="search-results" class="mt-4">
    <div class="initial-message">
      <p>Enter keywords to search blog posts...</p>
    </div>
  </div>
</div>

<!-- 更新为正确的脚本路径 -->
<script src="{{ '/assets/vendor/startbootstrap-clean-blog/js/search.js' | relative_url }}"></script>