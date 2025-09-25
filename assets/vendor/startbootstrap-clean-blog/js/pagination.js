// 分页配置
const POSTS_PER_PAGE = 5;

// 初始化分页功能
function initPagination() {
    // 检查数据是否已加载
    if (!window.allPosts || window.allPosts.length === 0) {
        console.error("文章数据未加载！");
        document.getElementById('posts-container').innerHTML = '<p class="text-muted text-center py-5">No posts available</p>';
        document.getElementById('pagination').style.display = 'none';
        return;
    }
    
    const postsContainer = document.getElementById('posts-container');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const currentPageSpan = document.getElementById('current-page');
    const totalPagesSpan = document.getElementById('total-pages');
    
    // 获取当前分类
    const currentCategory = window.currentCategory;
    
    // ✅ 修正：根据分类过滤文章，并去除重复
    let filteredPosts = window.allPosts;
    if (currentCategory) {
        filteredPosts = window.allPosts.filter(post => {
            return post.categories && post.categories.includes(currentCategory);
        });
    }
    
    // ✅ 新增：去除重复文章（基于URL去重）
    const uniquePosts = [];
    const seenUrls = new Set();
    
    filteredPosts.forEach(post => {
        if (!seenUrls.has(post.url)) {
            seenUrls.add(post.url);
            uniquePosts.push(post);
        }
    });
    
    filteredPosts = uniquePosts;
    
    const totalPages = Math.ceil(filteredPosts.length / POSTS_PER_PAGE);
    totalPagesSpan.textContent = totalPages;
    
    let currentPage = 1;
    
    function renderPosts() {
        postsContainer.innerHTML = '';
        
        // 如果没有文章
        if (filteredPosts.length === 0) {
            postsContainer.innerHTML = `
                <div class="no-posts text-center py-5">
                    <h3 class="text-muted">暂无文章</h3>
                    <p class="text-muted">当前分类还没有发布任何文章。</p>
                </div>
            `;
            document.getElementById('pagination').style.display = 'none';
            return;
        }
        
        const startIndex = (currentPage - 1) * POSTS_PER_PAGE;
        const endIndex = Math.min(startIndex + POSTS_PER_PAGE, filteredPosts.length);
        
        for (let i = startIndex; i < endIndex; i++) {
            const post = filteredPosts[i];
            const postElement = document.createElement('article');
            postElement.className = 'post-preview';
            
            // ✅ 修正：删除分类标签显示，只保留标题、副标题和元数据
            postElement.innerHTML = `
                <a href="${post.url}">
                    <h2 class="post-title">${post.title}</h2>
                    <h3 class="post-subtitle">${post.subtitle || ''}</h3>
                </a>
                <!-- ❌ 删除分类标签：<div class="post-category">${post.categories && post.categories.length > 0 ? post.categories[0] : ''}</div> -->
                <p class="post-meta">
                    Posted by ${post.author} on ${post.date} &middot; ${post.read_time}
                </p>
            `;
            
            postsContainer.appendChild(postElement);
            
            // 添加分隔线（除了最后一篇文章）
            if (i < endIndex - 1) {
                const hr = document.createElement('hr');
                postsContainer.appendChild(hr);
            }
        }
    }
    
    function updatePagination() {
        currentPageSpan.textContent = currentPage;
        prevBtn.disabled = currentPage === 1;
        nextBtn.disabled = currentPage === totalPages;
        
        // 更新按钮样式
        prevBtn.classList.toggle('disabled', currentPage === 1);
        nextBtn.classList.toggle('disabled', currentPage === totalPages);
        
        // 如果没有文章或只有一页，隐藏分页控件
        if (filteredPosts.length <= POSTS_PER_PAGE) {
            document.getElementById('pagination').style.display = 'none';
        } else {
            document.getElementById('pagination').style.display = 'flex';
        }
    }
    
    function initPaginationControls() {
        renderPosts();
        updatePagination();
        
        prevBtn.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                renderPosts();
                updatePagination();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
        
        nextBtn.addEventListener('click', () => {
            if (currentPage < totalPages) {
                currentPage++;
                renderPosts();
                updatePagination();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
    }
    
    initPaginationControls();
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', initPagination);