document.addEventListener('DOMContentLoaded', function() {
    console.log("=== 搜索功能初始化 ===");
    
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const resultsContainer = document.getElementById('search-results');
    let searchData = [];
    
    // 检查元素是否存在
    if (!searchInput || !searchButton || !resultsContainer) {
        console.error("❌ 搜索元素未找到:");
        console.error("- search-input:", document.getElementById('search-input'));
        console.error("- search-button:", document.getElementById('search-button'));
        console.error("- search-results:", document.getElementById('search-results'));
        return;
    }
    
    console.log("✅ 搜索元素已找到");
    
    // 加载搜索数据
    fetch('/search.json')
        .then(response => {
            console.log("📊 搜索数据响应状态:", response.status);
            if (!response.ok) {
                throw new Error(`HTTP错误! 状态: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("✅ 搜索数据加载成功:", data.length, "篇文章");
            console.log("📝 文章列表:", data.map(post => post.title));
            searchData = data;
        })
        .catch(error => {
            console.error('❌ 加载搜索数据失败:', error);
            resultsContainer.innerHTML = '<p class="text-danger">加载搜索数据失败，请刷新页面重试。</p>';
        });
    
    // 搜索功能
    function performSearch(query) {
        console.log("🔍 开始搜索:", query);
        console.log("📚 可用文章数量:", searchData.length);
        
        if (!query.trim()) {
            resultsContainer.innerHTML = '<div class="initial-message"><p>请输入关键词搜索文章...</p></div>';
            return;
        }
        
        if (searchData.length === 0) {
            resultsContainer.innerHTML = '<p class="text-warning">搜索数据尚未加载完成，请稍后重试。</p>';
            return;
        }
        
        const startTime = performance.now();
        const keywords = query.toLowerCase().split(/\s+/).filter(k => k);
        
        console.log("🔑 搜索关键词:", keywords);
        
        // 搜索逻辑
        const results = searchData.map(post => {
            let score = 0;
            const content = (post.content || '').toLowerCase();
            const title = (post.title || '').toLowerCase();
            const excerpt = (post.excerpt || '').toLowerCase();
            
            console.log(`📖 检查文章: "${post.title}"`);
            
            // 标题完全匹配（最高权重）
            if (title === query.toLowerCase()) {
                score += 100;
                console.log("🎯 标题完全匹配!");
            }
            
            // 标题包含匹配
            keywords.forEach(keyword => {
                if (title.includes(keyword)) {
                    score += 10;
                    console.log("📌 标题包含关键词:", keyword);
                }
            });
            
            // 内容匹配
            keywords.forEach(keyword => {
                const contentMatches = content.split(keyword).length - 1;
                if (contentMatches > 0) {
                    score += contentMatches;
                    console.log("📄 内容匹配关键词:", keyword, "出现次数:", contentMatches);
                }
            });
            
            // 标签匹配
            if (post.tags && Array.isArray(post.tags)) {
                keywords.forEach(keyword => {
                    if (post.tags.some(tag => 
                        tag.toLowerCase().includes(keyword)
                    )) {
                        score += 5;
                        console.log("🏷️ 标签匹配:", keyword);
                    }
                });
            }
            
            // 分类匹配
            if (post.categories && Array.isArray(post.categories)) {
                keywords.forEach(keyword => {
                    if (post.categories.some(cat => 
                        cat.toLowerCase().includes(keyword)
                    )) {
                        score += 3;
                        console.log("📂 分类匹配:", keyword);
                    }
                });
            }
            
            console.log(`📊 文章 "${post.title}" 得分: ${score}`);
            return { post, score };
        })
        .filter(item => {
            const hasMatch = item.score > 0;
            console.log(`📋 文章 "${item.post.title}" ${hasMatch ? '有匹配' : '无匹配'}`);
            return hasMatch;
        })
        .sort((a, b) => b.score - a.score);
        
        const endTime = performance.now();
        const searchTime = (endTime - startTime).toFixed(2);
        
        console.log("📈 搜索结果:", results.length, "个匹配");
        console.log("⏱️ 搜索耗时:", searchTime, "ms");
        
        displayResults(results, query, searchTime);
    }
    
    // 显示结果
    function displayResults(results, query, searchTime) {
        if (results.length === 0) {
            console.log("❌ 无搜索结果");
            
            // 显示所有可用文章标题用于调试
            const allTitles = searchData.map(post => post.title).join(', ');
            console.log("📚 所有可用文章:", allTitles);
            
            resultsContainer.innerHTML = `
                <div class="no-results">
                    <h3>未找到相关文章</h3>
                    <p>搜索 "<strong>${query}</strong>" 没有找到匹配的文章。</p>
                    <div class="debug-info">
                        <details>
                            <summary>调试信息（点击展开）</summary>
                            <p>可用文章: ${searchData.length} 篇</p>
                            <p>文章标题列表: ${allTitles}</p>
                        </details>
                    </div>
                    <p>建议：</p>
                    <ul>
                        <li>检查拼写是否正确</li>
                        <li>尝试使用更短的关键词</li>
                        <li>尝试使用文章的部分标题</li>
                    </ul>
                </div>
            `;
            return;
        }
        
        console.log("✅ 显示搜索结果");
        
        let html = `
            <div class="search-stats mb-3">
                <p>找到 ${results.length} 个结果，耗时 ${searchTime}ms</p>
                <p>搜索词: <strong>"${query}"</strong></p>
            </div>
            <div class="results-list">
        `;
        
        results.forEach((result, index) => {
            const post = result.post;
            console.log(`📄 显示结果 ${index + 1}: ${post.title}`);
            
            html += `
                <article class="search-result">
                    <h3><a href="${post.url}">${post.title}</a></h3>
                    <div class="post-meta">
                        <span class="date">${post.date}</span>
                        ${post.categories ? `<span class="categories">${post.categories.join(', ')}</span>` : ''}
                        <span class="score">匹配度: ${result.score}分</span>
                    </div>
                    <p class="excerpt">${post.excerpt || post.content.substring(0, 150)}...</p>
                </article>
            `;
        });
        
        html += '</div>';
        resultsContainer.innerHTML = html;
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
    
    // 实时搜索（可选）
    let searchTimeout;
    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            performSearch(this.value);
        }, 500);
    });
    
    console.log("✅ 搜索功能初始化完成");
});