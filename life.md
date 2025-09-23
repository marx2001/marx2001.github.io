---
layout: page          # 使用 page 布局模板
title: Life        # 页面标题
permalink: /life/  # URL 地址
background: /img/bg-life.jpg
---
{% assign life = site.posts | where:"categories","life" %}
{% for post in life limit:5 %}

<article class="post-preview">
  <a href="{{ post.url | prepend: site.baseurl | replace: '//', '/' }}">
    <h2 class="post-title">{{ post.title }}</h2>
    {% if post.subtitle %}
    <h3 class="post-subtitle">{{ post.subtitle }}</h3>
    {% else %}
    <h3 class="post-subtitle">{{ post.excerpt | strip_html | truncatewords: 15 }}</h3>
    {% endif %}
  </a>
  <p class="post-meta">Posted by
    {% if post.author %}
    {{ post.author }}
    {% else %}
    {{ site.author }}
    {% endif %}
    on
    {{ post.date | date: '%B %d, %Y' }} &middot; {% include read_time.html content=post.content %}            
  </p>
</article>

<hr>


{% endfor %}
