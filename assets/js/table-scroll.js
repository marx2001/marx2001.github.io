/* 让正文里的宽表格在窄屏独立横向滚动，而不是撑破正文列。
   不设置 min-width：内容放得下时正常铺满，放不下时才出现横向滚动。
   与作者手工包装的 .table-scroll 共存（已包装的表格跳过）。 */
document.addEventListener("DOMContentLoaded", function () {
  var tables = document.querySelectorAll(".post-content table");
  for (var i = 0; i < tables.length; i += 1) {
    var table = tables[i];
    var parent = table.parentElement;
    if (parent && (parent.classList.contains("table-scroll-x") || parent.classList.contains("table-scroll"))) {
      continue;
    }
    var wrapper = document.createElement("div");
    wrapper.className = "table-scroll-x";
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  }
});
