#!/bin/sh

vocabs=30000


./crawler/daily_reuters.py

./main.py -vocabs ${vocabs} -epochs 500 -static False
./main.py -vocabs ${vocabs} -eval True
./main.py -vocabs ${vocabs} -predict "Top executive behind Baidu's artificial intelligence drive steps aside"

