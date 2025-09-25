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
    
    // 获取当前分类（从页面变量获取）
    const currentCategory = window.currentCategory;
    console.log('当前分类:', currentCategory);
    
    // ✅ 修正：根据分类标签正确筛选文章
    let filteredPosts = window.allPosts;
    if (currentCategory) {
        filteredPosts = window.allPosts.filter(post => {
            // 检查文章的分类标签是否包含当前分类
            if (!post.categories || post.categories.length === 0) {
                return false;
            }
            
            // 将分类标签转换为小写进行比较（不区分大小写）
            const postCategories = post.categories.map(cat => cat.toLowerCase());
            const targetCategory = currentCategory.toLowerCase();
            
            return postCategories.includes(targetCategory);
        });
    }
    
    console.log('过滤后的文章数量:', filteredPosts.length);
    console.log('所有文章数量:', window.allPosts.length);
    
    const totalPages = Math.ceil(filteredPosts.length / POSTS_PER_PAGE);
    totalPagesSpan.textContent = totalPages;
    
    let currentPage = 1;
    
    function renderPosts() {
        postsContainer.innerHTML = '';
        
        // ✅ 修正：如果没有文章，显示友好提示
        if (filteredPosts.length === 0) {
            postsContainer.innerHTML = `
                <div class="no-posts text-center py-5">
                    <h3 class="text-muted">暂无文章</h3>
                    <p class="text-muted">当前分类还没有发布任何文章。</p>
                    <p class="text-muted small">分类: ${currentCategory || '所有文章'}</p>
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
            
            // ✅ 修正：使用文章的第一个分类作为显示标签
            const displayCategory = post.categories && post.categories.length > 0 ? 
                post.categories[0] : '未分类';
            
            postElement.innerHTML = `
                <a href="${post.url}">
                    <h2 class="post-title">${post.title}</h2>
                    <h3 class="post-subtitle">${post.subtitle || ''}</h3>
                </a>
                <div class="post-category">${displayCategory}</div>
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
        
        // 更新按钮状态
        const isFirstPage = currentPage === 1;
        const isLastPage = currentPage === totalPages;
        
        prevBtn.disabled = isFirstPage;
        nextBtn.disabled = isLastPage;
        
        prevBtn.classList.toggle('disabled', isFirstPage);
        nextBtn.classList.toggle('disabled', isLastPage);
        
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