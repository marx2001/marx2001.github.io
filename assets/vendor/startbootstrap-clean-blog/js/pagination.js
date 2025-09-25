// 分页配置
const POSTS_PER_PAGE = 5;
let currentPage = 1;

document.addEventListener('DOMContentLoaded', function() {
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
    
    function renderPosts() {
        postsContainer.innerHTML = '';
        
        const startIndex = (currentPage - 1) * POSTS_PER_PAGE;
        const endIndex = Math.min(startIndex + POSTS_PER_PAGE, window.allPosts.length);
        
        for (let i = startIndex; i < endIndex; i++) {
            const post = window.allPosts[i];
            const postElement = document.createElement('article');
            postElement.className = 'post-preview';
            
            // 正确显示作者信息
            const author = post.author || window.siteAuthor || 'Unknown Author';
            
            postElement.innerHTML = `
                <a href="${post.url}">
                    <h2 class="post-title">${post.title}</h2>
                    <h3 class="post-subtitle">${post.excerpt || ''}</h3>
                </a>
                <p class="post-meta">Posted by ${author} on ${post.date}</p>
            `;
            postsContainer.appendChild(postElement);
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
    
    function initPagination() {
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
    
    initPagination();
});