#!/bin/bash

awk -F',' '{print $1}' ./input/news_bloomberg* | sort | uniq > ./input/finished.list
