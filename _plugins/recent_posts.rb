require 'time'

module Jekyll
  class RecentPosts < Generator
    safe true
    priority :low

    def generate(site)
      # An explicit update date takes precedence; older posts use their publication date.
      site.data['recent_posts'] = site.posts.docs.sort_by do |post|
        updated = post.data['last_modified_at'] || post.date
        [Time.parse(updated.to_s), post.date, post.url]
      end.reverse.first(site.config.dig('pagination', 'per_page') || 5)
    end
  end
end
