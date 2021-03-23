import requests
import os.path
from bs4 import BeautifulSoup as BSoup

ngs_url = "https://ngs.ru"


def crawl_problems():
    session = requests.Session()
    problem_url = ngs_url + "/text/format/Проблема/"
    page = "?page="
    count = 1

    for i in range(1, 16):  # TODO: fix to any page count
        cur_url = problem_url + page + i.__str__()
        content = session.get(cur_url).content
        soup = BSoup(content, "html.parser")
        news = soup.find_all('article')
        for article in news:
            article_refs = article.find_all('a')
            tags = []
            for ref in article_refs:
                if ref.has_attr('slot') and ref.get('slot') == 'rubrics':
                    title = ref.get('title')
                    tags.append(title)
            main = article_refs[0]
            href = main.get('href')
            if "https" not in href:
                link = ngs_url + href
            else:
                if 'text' not in href:
                    continue
                else:
                    link = href
            file_name = f"data/{make_file_name(link, ngs_url)}"
            save_new_article(count, file_name, session, link, tags)
            count += 1


def make_file_name(full_link, site_prefix):
    without_prefix = full_link.replace(site_prefix, '')
    if 'text' in without_prefix:
        without_prefix = without_prefix.replace('text', '')
    link_parts = list(filter(None, without_prefix.split('/')))
    return '-'.join(map(str, link_parts)) + ".txt"


def save_new_article(count, file_name, session, url, tags):
    if not os.path.isfile(file_name):
        print(f'{count}. {url}')
        with open(file_name, "w", encoding='utf-8') as f:
            get_article(session, url, f, tags)
    else:
        print(f'{count}. File {file_name} exists')


def get_article(session, url, file, tags):
    if tags:
        file.write(' | '.join(map(str, tags)) + "\n")
    content = session.get(url).content
    soup = BSoup(content, "html.parser")
    headline = soup.find('h2', itemprop="headline")
    file.write(headline.get_text() + "\n")
    alternative_headline = soup.find('p', itemprop="alternativeHeadline")
    file.write(alternative_headline.get_text() + "\n")
    article_body = soup.find('div', itemprop="articleBody").find('div')
    for paragraphs in article_body.find_all(['div', 'p'], recursive=False):
        for part in paragraphs.find_all(['p', 'span'], recursive=True):
            file.write(part.get_text() + "\n")


def crawl_rss():
    session = requests.Session()
    rss_url = ngs_url + "/rss/"
    content = session.get(rss_url).content
    soup = BSoup(content, "html.parser")
    i = 1
    for item in soup.find_all('item', recursive=True):
        link = item.find("pdalink").text
        category = item.find("category").text
        file_name = f"data/rss/{make_file_name(link, ngs_url)}"
        tags = [category]
        save_new_article(i, file_name, session, link, tags)
        i += 1


if __name__ == '__main__':
    crawl_problems()
    # crawl_rss()
