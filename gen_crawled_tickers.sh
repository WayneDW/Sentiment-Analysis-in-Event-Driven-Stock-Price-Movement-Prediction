#!/bin/bash

awk -F',' '{print $1}' ./input/news_bloomberg.csv | uniq >> ./input/finished.list
