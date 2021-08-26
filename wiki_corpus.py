import re

import wikipedia
import os, json
import time

DUMP_DIR = "data"

root_page = "Indian independence movement"

current_dump_dir = os.path.abspath(os.curdir) + os.path.sep + DUMP_DIR
if not os.path.exists(current_dump_dir):
    os.mkdir(current_dump_dir)


def preprocess_page(page, title):
    if not page.title or page.title != title:
        page_title = title
    else:
        page_title = page.title
    empty_section = "\={2,3}.*\={2,3}\n{2,}\s*"
    page_content = re.sub(empty_section, "", page.content.lower())
    trailing_empty = "\={2,3}.*\={2,3}$"
    page_content = re.sub(trailing_empty, "", page_content)
    page_details = (page_title, page.links, page.categories, page_content)
    return page_details


def save_content(page_title, page_links, page_categories, page_content):
    filename = f"{current_dump_dir}{os.path.sep}{page_title}.txt"
    with open(filename, "w") as f:
        f.write(page_content)
    metafile = f"{current_dump_dir}/crawl_meta.json"
    if os.path.exists(metafile) and os.path.getsize(metafile) > 0:
        with open(metafile, "r") as f:
            json_data = json.loads(f.read())
    else:
        json_data = {}
    json_data[page_title] = {'links': page_links, 'categories': page_categories}
    with open(metafile, "w") as f:
        f.write(json.dumps(json_data))


def get_page(title, crawl=True):
    page = None
    try:
        print(f'crawling page->{title}')
        page = wikipedia.wikipedia.WikipediaPage(title)
    except wikipedia.exceptions.PageError as e:
        print(f"could not parse {e}")
        return
    page_title, page_links, page_categories, page_content = preprocess_page(page, title)
    save_content(page_title, page_links, page_categories, page_content)
    if crawl is True:
        print(f"crawling leaves of: {title} \n{60*'='}")
        for each in page.links:
            time.sleep(1)
            page_link = f"{current_dump_dir}/{each}.txt"
            if os.path.exists(page_link):
                print(f'crawl already exists: {page_link}', )
            else:
                # print(f'recursing leaf item ->{each}')
                get_page(each, False)


def interospect_crawl(display_string):
    print(display_string)
    try:
        crawl_metadata = open(f"{current_dump_dir}{os.path.sep}crawl_meta.json")
        json_object = json.load(crawl_metadata)
    except json.decoder.JSONDecodeError or FileNotFoundError:
        print('crawl metadata file not found')
        return

    detected_leaves = set(json_object[root_page]['links'])
    crawled_titles = list(json_object.keys())
    crawled_titles.pop(crawled_titles.index(root_page))
    crawled_leaves = set(crawled_titles)
    crawl_miss = list(detected_leaves.difference(crawled_leaves))
    crawl_miss.sort()
    if crawl_miss:
        print(f"missed crawling:\n{crawl_miss}")
        return crawl_miss


if __name__ == "__main__":
    print(f"dumping crawled pages at->{current_dump_dir}")
    crawl_miss = interospect_crawl("crawl stats before refresh")
    if crawl_miss:
        for each_links in crawl_miss:
            get_page(each_links, False)
    else:
        get_page(root_page)

