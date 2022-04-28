# Webpage for the paper

[Template live demo](https://mahdifarvardin.github.io/MTVital)

## Building the website locally

Follow [these instructions](https://help.github.com/articles/setting-up-your-github-pages-site-locally-with-jekyll/#requirements)
to install Ruby and Bundler.

Install the other plugins with

``` bash
$ bundle install
```

Edit `_config.yml` so that `baseurl` is an empty string, and launch the local
server with

``` bash
$ bundle exec jekyll serve -w
```

You can preview the webpage locally at

```
http://localhost:4000
```

## Configuring the template

All necessary configuration tweaks are done through `_config.yml`, which is
self-documented.

## Final step

Once you're ready, edit `_config.yml` so that `baseurl` is `"/repository-name"`,
commit, and push to Github.
