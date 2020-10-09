---
title: "Data Cleaning with Python and Pandas"
date: 2020-10-09
tags: [data cleaning, data science, messy data, pandas]
header: 
  image: 
excerpt: "Data Cleaning with Python and Pandas"
mathjax: "true"
---

# Data Cleaning with Python and Pandas

## Description
Data Scientists can spend up to 80% of their time cleaning data but it is necessary work to ensure a good functioning machine learning model. 

I'll be working with a small real estate dataset and cleaning it of various inconsistencies. Although the dataset is small it demonstrates a lot of real-world examples Data Scientists encounter on a regular basis.

## Raw Data
Below is a table of the raw data as it's stored in the csv.

|    PID    | ST_NUM | ST_NAME    | OWN_OCCUPIED | NUM_BEDROOMS | NUM_BATH | SQ_FT |
|-----------|--------|------------|--------------|--------------|----------|-------|
| 100001000 | 104    | pUTNAM     | Y            | 3            | 1        | 1000  |
| 100002000 | 197    | LEXINGTON  | N            | 3            | 1.5      | --    |
| 100003000 |        | LEXINGTON  | N            | n/a          | 1        | 850   |
| 100004000 | 201    | BERKELEY   | 12           | 1            | NaN      | 700   |
|           | 203    | BERKELEY   | Y            | 3            | 2        | 1600  |
| 100006000 | 207    | berkeley   | Y            | NA           | 1        | 800   |
| 100007000 | NA     | WASHINGTON |              | 2            | HURLEY   | 950   |
| 100008000 | 213    | TREMONT    | Y            | --           | 1        |       |
| 100009000 | 215    | TREMONT    | Y            | na           | 2        | 1800  |

Let's start by asking the following questions:
- What are the features?
- What are the expected types (int, float, string, boolean)?
- Are there obvious missing data values (values that Pandas can detect)?
- Are there other types of missing data that's not so obvious (can't easily detect with Pandas)?

## Loading Data

We'll first start out by getting a quick feel for the data by looking at the first few rows.


```python
# Importing libraries
import pandas as pd
import numpy as np

# Read csv file into a pandas dataframe
df = pd.read_csv("property data.csv")

# Take a look at the data
df
```
