document.addEventListener('DOMContentLoaded', function() {
    console.log("Search initialized");
    
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const resultsContainer = document.getElementById('search-results');
    let searchData = [];
    
    // 元素存在性检查
    if (!searchInput || !searchButton || !resultsContainer) {
        console.error("Required elements not found");
        return;
    }
    
    // 显示路径信息
    console.log("Search script path:", window.location.origin + '/assets/vendor/startbootstrap-clean-blog/js/search.js');
    console.log("Search data path:", window.location.origin + '/search.json');
    
    // 加载搜索数据
    fetch('/search.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Search data loaded:", data.length, "posts");
            searchData = data;
        })
        .catch(error => {
            console.error('Error loading search data:', error);
            resultsContainer.innerHTML = '<p class="text-danger">Failed to load search data. Please try again later.</p>';
        });
    
    // 搜索功能
    function performSearch(query) {
        console.log("Searching for:", query);
        
        // 清除结果容器
        resultsContainer.innerHTML = '<div class="loader"><i class="fas fa-spinner fa-spin"></i> Searching...</div>';
        
        if (!query.trim()) {
            resultsContainer.innerHTML = '<div class="initial-message"><p>Enter keywords to search blog posts...</p></div>';
            return;
        }
        
        const startTime = performance.now();
        const keywords = query.toLowerCase().split(/\s+/).filter(k => k);
        
        // 搜索逻辑
        const results = searchData.map(post => {
            let score = 0;
            const content = (post.content || '').toLowerCase();
            const title = (post.title || '').toLowerCase();
            const excerpt = (post.excerpt || '').toLowerCase();
            
            // 标题匹配
            keywords.forEach(keyword => {
                if (title.includes(keyword)) score += 5;
            });
            
            // 内容匹配
            keywords.forEach(keyword => {
                if (content.includes(keyword)) score += 1;
            });
            
            // 标签匹配
            if (post.tags && Array.isArray(post.tags)) {
                keywords.forEach(keyword => {
                    if (post.tags.some(tag => 
                        tag.toLowerCase().includes(keyword)
                    )) score += 3;
                });
            }
            
            // 分类匹配
            if (post.categories && Array.isArray(post.categories)) {
                keywords.forEach(keyword => {
                    if (post.categories.some(cat => 
                        cat.toLowerCase().includes(keyword)
                    )) score += 2;
                });
            }
            
            return { post, score };
        })
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score);
        
        const endTime = performance.now();
        const searchTime = (endTime - startTime).toFixed(2);
        
        displayResults(results, query, searchTime);
    }
    
    // 显示结果
    function displayResults(results, query, searchTime) {
        if (results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="no-results">
                    <h3>No results found</h3>
                    <p>Your search for <strong>"${query}"</strong> did not match any posts.</p>
                    <p>Suggestions:</p>
                    <ul>
                        <li>Try different keywords</li>
                        <li>Try more general keywords</li>
                        <li>Check your spelling</li>
                    </ul>
                </div>
            `;
            return;
        }
        
        let html = `
            <div class="search-stats mb-3">
                <p>Found ${results.length} results in ${searchTime}ms for <strong>"${query}"</strong></p>
            </div>
            <div class="results-list">
        `;
        
        results.forEach(result => {
            const post = result.post;
            const excerpt = highlightKeywords(
                post.excerpt || post.content.substring(0, 200), 
                query
            );
            
            html += `
                <article class="search-result">
                    <h3><a href="${post.url}">${highlightKeywords(post.title, query)}</a></h3>
                    <div class="post-meta">
                        <span class="date">${post.date}</span>
                        ${post.categories ? `<span class="categories">${post.categories.join(', ')}</span>` : ''}
                    </div>
                    <p class="excerpt">${excerpt}...</p>
                    <div class="match-score">
                        <small>Relevance: ${Math.round(result.score * 10)}%</small>
                    </div>
                </article>
            `;
        });
        
        html += '</div>';
        resultsContainer.innerHTML = html;
    }
    
    // 高亮关键词
    function highlightKeywords(text, query) {
        if (!text) return '';
        
        const keywords = query.toLowerCase().split(/\s+/).filter(k => k);
        let highlighted = text;
        
        keywords.forEach(keyword => {
            const regex = new RegExp(`(${escapeRegExp(keyword)})`, 'gi');
            highlighted = highlighted.replace(regex, '<mark>$1</mark>');
        });
        
        return highlighted;
    }
    
    // 转义正则特殊字符
    function escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    
    // 事件监听
    searchInput.addEventListener('keyup', function(e) {
        if (e.key === 'Enter') {
            performSearch(this.value);
        }
    });
    
    searchButton.addEventListener('click', function() {
        performSearch(searchInput.value);
    });
    
    // 添加输入监听器
    searchInput.addEventListener('input', function() {
        if (this.value.trim() === '') {
            resultsContainer.innerHTML = '<div class="initial-message"><p>Enter keywords to search blog posts...</p></div>';
        }
    });
});