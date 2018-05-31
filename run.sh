#!/bin/sh

vocabs=10000
#./tokenize_news.py -vocabs $vocabs
#./main.py -vocabs $vocabs -epochs 5
#./main.py -vocabs $vocabs -epochs 5 -predict True 
./main.py -vocabs $vocabs -predict True -snapshot 2018-05-30_23-20-18/best_steps_1.pt -date 20180524
./main.py -vocabs $vocabs -predict True -snapshot 2018-05-30_23-20-18/best_steps_1.pt -date 20180525
./main.py -vocabs $vocabs -predict True -snapshot 2018-05-30_23-20-18/best_steps_1.pt -date 20180526


