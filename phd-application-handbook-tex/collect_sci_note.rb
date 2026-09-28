#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "jekyll"

root = File.expand_path("..", __dir__)
config = Jekyll.configuration(
  "source" => root,
  "destination" => File.join(root, "_site"),
  "quiet" => true,
  "plugins_dir" => [],
  "disable_disk_cache" => true
)

site = Jekyll::Site.new(config)
site.reset
site.read
converter = site.find_converter_instance(Jekyll::Converters::Markdown)

posts = site.posts.docs.filter_map do |post|
  categories = Array(post.data["categories"]).map(&:to_s)
  next unless categories.include?("sci-note")

  {
    "source_path" => post.relative_path.sub(%r{\A/}, ""),
    "title" => post.data.fetch("title", File.basename(post.path, ".md")).to_s,
    "subtitle" => post.data.fetch("subtitle", "").to_s,
    "date" => post.date.strftime("%Y-%m-%d"),
    "url" => post.url.to_s,
    "categories" => categories,
    "html" => converter.convert(post.content)
  }
end

posts.sort_by! { |post| [post.fetch("date"), post.fetch("source_path")] }
STDOUT.write(JSON.generate(posts))
