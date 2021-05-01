#!/bin/bash

# analysis/csvにリネームされたcsvファイルが存在すると仮定
{
    echo "datetime,open,high,low,close,volume"
    {
        cat USDJPY_2015_all.csv
        cat USDJPY_2016_all.csv
        cat USDJPY_2017_all.csv
        cat USDJPY_2018_all.csv
        cat USDJPY_2019_all.csv
        cat USDJPY_2020_all.csv
    } | grep -v "^\s*$" | sed -E -e "s|(^\w*)\.(\w*)\.(\w*),|\1-\2-\3-|"
} > concat_USDJPY2015_2020.csv
