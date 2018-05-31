#!/bin/sh

vocabs=10000


./crawler/daily_reuters.py

./main.py -vocabs 10000 -epochs 100 -t 100000
./main.py -vocabs 10000 -eval True
./main.py -vocabs 10000 -predict "disaster trump some scandal happens"

