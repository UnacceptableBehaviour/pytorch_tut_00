#! /usr/bin/env python

# fetch transforms list
# https://pytorch.org/vision/stable/transforms.html

# source
# https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python

# from bs4 import BeautifulSoup
# my_HTML = #Some HTML file (could be a website, you can use urllib for that)
# soup = BeautifulSoup(my_HTML, 'html.parser')
# print(soup.prettify())

# pip install beautifulsoup4
# or
# conda install -c anaconda beautifulsoup4

import re
from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "https://pytorch.org/vision/stable/transforms.html"
html = urlopen(url).read()
soup = BeautifulSoup(html, features="html.parser")

# kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # rip it out

# get text
text = soup.get_text()

# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)

# find
# class torchvision.transforms.functional.InterpolationMode[source]
# r/class torchvision.transforms.(.*?)\[source\]/

matches = re.findall("class torchvision.transforms.(.*?)\[source\]", text)

for i in matches:
  print(i)

#
# matches = re.findall("torchvision.transforms.functional.", text)
#
# for i in matches:
#   print(i)
