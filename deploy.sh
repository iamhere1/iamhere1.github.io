rm -rf public/_posts
mkdir -p public/_posts/
cp -rf source/_posts/*md public/_posts/
hexo g
hexo d
