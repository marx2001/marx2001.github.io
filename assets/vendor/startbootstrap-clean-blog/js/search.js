document.addEventListener('DOMContentLoaded', function() {
    console.log("Search script loaded from: assets/vendor/startbootstrap-clean-blog/js/search.js");
    
    // 获取DOM元素
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const resultsContainer = document.getElementById('search-results');
    
    // 检查元素是否存在
    if (!searchInput || !searchButton || !resultsContainer) {
        console.error("Required elements not found");
        return;
    }
    
    // 存储所有文章数据
    let allPosts = [];
    
    // 加载文章数据
    fetch('/search.json')
        .then(response => {
            console.log("Fetching search.json...");
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(posts => {
            console.log(`Loaded ${posts.length} posts from search.json`);
            allPosts = posts;
        })
        .catch(error => {
            console.error('Error loading search data:', error);
            resultsContainer.innerHTML = '<p class="text-danger">Failed to load search data. Please try again later.</p>';
        });
    
    // 执行搜索
    function performSearch() {
        const query = searchInput.value.trim();
        
        // 清空结果容器
        resultsContainer.innerHTML = '<div class="loader"><i class="fas fa-spinner fa-spin"></i> Searching...</div>';
        
        if (!query) {
            resultsContainer.innerHTML = '<div class="initial-message"><p>Enter keywords to search blog posts.</p></div>';
            return;
        }
        
        if (allPosts.length === 0) {
            resultsContainer.innerHTML = '<p class="text-warning">Search data not loaded yet. Please wait.</p>';
            return;
        }
        
        const startTime = performance.now();
        const queryLower = query.toLowerCase();
        
        // 搜索逻辑
        const results = allPosts.filter(post => {
            // 检查标题
            if (post.title && post.title.toLowerCase().includes(queryLower)) return true;
            
            // 检查内容
            if (post.content && post.content.toLowerCase().includes(queryLower)) return true;
            
            // 检查摘要
            if (post.excerpt && post.excerpt.toLowerCase().includes(queryLower)) return true;
            
            // 检查标签
            if (post.tags && Array.isArray(post.tags)) {
                for (const tag of post.tags) {
                    if (tag.toLowerCase().includes(queryLower)) return true;
                }
            }
            
            // 检查分类
            if (post.categories && Array.isArray(post.categories)) {
                for (const category of post.categories) {
                    if (category.toLowerCase().includes(queryLower)) return true;
                }
            }
            
            return false;
        });
        
        const endTime = performance.now();
        const searchTime = (endTime - startTime).toFixed(2);
        
        // 显示结果
        displayResults(results, query, searchTime);
    }
    
    // 显示搜索结果
    function displayResults(results, query, searchTime) {
        if (results.length === 0) {
            // 显示所有可用文章标题用于调试
            const allTitles = allPosts.map(post => post.title).join(', ');
            console.log("All available posts:", allTitles);
            
            resultsContainer.innerHTML = `
                <div class="no-results">
                    <h3>No results found</h3>
                    <p>Your search for <strong>"${query}"</strong> did not match any documents.</p>
                    <div class="debug-info">
                        <details>
                            <summary>Debug Information</summary>
                            <p>Available posts: ${allPosts.length}</p>
                            <p>Search time: ${searchTime}ms</p>
                            <p>All titles: ${allTitles}</p>
                        </details>
                    </div>
                    <p>Suggestions:</p>
                    <ul>
                        <li>Make sure all words are spelled correctly.</li>
                        <li>Try different keywords.</li>
                        <li>Try more general keywords.</li>
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
        
        results.forEach(post => {
            // 高亮标题中的关键词
            const title = highlightKeywords(post.title, query);
            // 截取一段内容
            const contentSnippet = post.content.substring(0, 200) + '...';
            const excerpt = highlightKeywords(contentSnippet, query);
            
            html += `
                <article class="search-result">
                    <h2><a href="${post.url}">${title}</a></h2>
                    <div class="post-meta">
                        <span class="date">${post.date}</span>
                        ${post.categories ? `<span class="categories">${post.categories.join(', ')}</span>` : ''}
                    </div>
                    <p class="excerpt">${excerpt}</p>
                </article>
            `;
        });
        
        html += '</div>';
        resultsContainer.innerHTML = html;
    }
    
    // 高亮关键词
    function highlightKeywords(text, query) {
        if (!text || !query) return text;
        
        const regex = new RegExp(query, 'gi');
        return text.replace(regex, match => `<mark>${match}</mark>`);
    }
    
    // 绑定事件
    searchInput.addEventListener('keyup', function(event) {
        if (event.key === 'Enter') {
            performSearch();
        }
    });
    
    searchButton.addEventListener('click', performSearch);
    
    // 添加输入监听器
    searchInput.addEventListener('input', function() {
        if (this.value.trim() === '') {
            resultsContainer.innerHTML = '<div class="initial-message"><p>Enter keywords to search blog posts.</p></div>';
        }
    });
});