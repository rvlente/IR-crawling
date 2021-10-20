FROM nils2142/archlinux_aur

RUN pacman -Syyu --noconfirm python python-pipenv 
RUN mkdir /WORK && cd /WORK
RUN cd WORK &&\
    pipenv --python 3.9 &&\ 
    pipenv install scrapy textblob polyglot BeautifulSoup4