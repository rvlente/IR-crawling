import time
import scrapy
from scrapy.http.response.html import HtmlResponse
from scrapy import signals
import textblob
import polyglot
from polyglot.detect import Detector
from bs4 import BeautifulSoup
import re
import json
import shutil
from dataclasses import dataclass
import os

SAVE_FILE = "cache/state.json"

URL_RE = re.compile("""https?:\/\/([^\/]*)\/?.*""")
DUTCH_URL = re.compile(""".*\Wnl(?:(?:\W.*)|$)""")

URL = "https://www.wikipedia.nl/"

def print_to_file(f, *args, **kwargs):
    with open(f, "a") as f_:
        print(*args, **kwargs, file=f_)

def extract_text(response: HtmlResponse) -> str:
    soup = BeautifulSoup(response.text)
    return soup.get_text()


def url_without_query(url_str: str):
    return url_str.split("?")[0]

@dataclass
class CrawlState:
    in_queue: set
    history: set


class Spider(scrapy.Spider):
    name = "wiki"

    def __init__(self, *args, **kwargs):
        self._save_every = 10.0
        self._last_save = time.time()
        super().__init__(*args, **kwargs)

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(Spider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider


    def spider_closed(self, spider):
        self.save(SAVE_FILE)


    def save(self, file_):
        if os.path.exists(file_):
            shutil.copy(file_, f"{file_}.old")

        state = dict(
            history=[i for i in self.history],
            in_queue=[i for i in self.in_queue],
        )

        with open(file_, "w") as f:
            json.dump(state, f)

    def load(self, file_) -> CrawlState:
        with open(file_, "r") as f:
            state = json.load(f)

        return CrawlState({u for u in state["in_queue"]}, {u for u in state["history"]})
        

    def start_requests(self):
        try:
            state = self.load(SAVE_FILE)
            self.history = state.history
            self.in_queue = state.in_queue
        except IOError:
            self.history = set()
            self.in_queue = {URL}

        for url in self.in_queue:
            yield scrapy.Request(url=url, callback=self.process_page)

    def is_dutch_webpage(self, response: HtmlResponse):
        # text = extract_text(response)

        # try: 
        #     lang = Detector(text).language.name
        #     return lang in ["Dutch"]

        # except polyglot.detect.base.UnknownLanguage:
        #     return True

        langs = [l for l in response.css("html::attr(lang)")]

        if len(langs) == 0:
            return self.is_dutch_url(response.url)

        for l in langs:

            while not isinstance(l, str):
                l = l.get()

            print_to_file("cache/langs", l)
            if "nl" in  l.lower():
                return True

        return False


    def is_dutch_url(self, url_str: str):
        try:
            match = URL_RE.match(url_str)
            url_base = match.group(1)
            # print_to_file("cache/fff", url_base)
            return DUTCH_URL.match(url_base) is not None
        except AttributeError:
            print_to_file("cache/fff", f"urlstr:\n{url_str}")
            print_to_file("cache/fff", "--------------------------")
            return DUTCH_URL.match(url_str) is not None

    def maybe_save(self):
        t = time.time()
        if t - self._last_save > self._save_every:
            self.save(SAVE_FILE)
            self._last_save = time.time()
        


    def process_page(self, response: HtmlResponse):
        self.maybe_save()

        if response.request.meta.get('redirect_urls'):
            url = response.request.meta['redirect_urls'][0]
            url_nq = url_without_query(response.url)

            if url_nq in self.history:
                self.in_queue.remove(url)
                return

            self.history.add(url_nq)
            
        else:
            url = response.request.url

        self.in_queue.remove(url)


        
        if not self.is_dutch_webpage(response):
            return

        print_to_file("cache/urls_visited", response.url)
        # print_to_file("tmpfile2", *self.in_queue)
        

        links = response.css("a::attr(href)")

        to_process = []

        for link in links:
            link_str: str = link.get()

            link_str = response.urljoin(link_str)
            link_str = link_str.split("?")[0]

            if link_str in self.history or not link_str.startswith("http"):
                continue

            
            self.history.add(link_str)

            # if self.is_dutch_url(link_str):
            to_process.append(link_str)
            

        # print_to_file("cache/tmpfile3", *to_process)
        self.in_queue.update(to_process)

        for link_str in to_process:
            yield scrapy.Request(link_str, callback=self.process_page)
            