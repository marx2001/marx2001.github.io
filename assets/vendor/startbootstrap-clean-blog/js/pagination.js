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
    
    const totalPages = Math.ceil(window.allPosts.length / POSTS_PER_PAGE);
    totalPagesSpan.textContent = totalPages;
    
    let currentPage = 1;
    
    function renderPosts() {
        postsContainer.innerHTML = '';
        
        const startIndex = (currentPage - 1) * POSTS_PER_PAGE;
        const endIndex = Math.min(startIndex + POSTS_PER_PAGE, window.allPosts.length);
        
        for (let i = startIndex; i < endIndex; i++) {
            const post = window.allPosts[i];
            const postElement = document.createElement('article');
            postElement.className = 'post-preview';
            
            // ✅ 根据图片格式生成文章内容
            // 图片显示格式：标题 + 分类 + 元数据
            const category = post.categories && post.categories.length > 0 ? 
                post.categories[0] : 'Uncategorized';
            
            postElement.innerHTML = `
                <a href="${post.url}">
                    <h2 class="post-title">${post.title}</h2>
                    <h3 class="post-subtitle">${post.subtitle || ''}</h3>
                </a>
                <div class="post-category">${category}</div>
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
    }
    
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

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', initPagination);