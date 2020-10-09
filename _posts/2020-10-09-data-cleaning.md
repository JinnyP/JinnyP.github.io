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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PID</th>
      <th>ST_NUM</th>
      <th>ST_NAME</th>
      <th>OWN_OCCUPIED</th>
      <th>NUM_BEDROOMS</th>
      <th>NUM_BATH</th>
      <th>SQ_FT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001000.0</td>
      <td>104.0</td>
      <td>pUTNAM</td>
      <td>Y</td>
      <td>3</td>
      <td>1</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002000.0</td>
      <td>197.0</td>
      <td>LEXINGTON</td>
      <td>N</td>
      <td>3</td>
      <td>1.5</td>
      <td>--</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003000.0</td>
      <td>NaN</td>
      <td>LEXINGTON</td>
      <td>N</td>
      <td>NaN</td>
      <td>1</td>
      <td>850</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004000.0</td>
      <td>201.0</td>
      <td>BERKELEY</td>
      <td>12</td>
      <td>1</td>
      <td>NaN</td>
      <td>700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>203.0</td>
      <td>BERKELEY</td>
      <td>Y</td>
      <td>3</td>
      <td>2</td>
      <td>1600</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100006000.0</td>
      <td>207.0</td>
      <td>berkeley</td>
      <td>Y</td>
      <td>NaN</td>
      <td>1</td>
      <td>800</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100007000.0</td>
      <td>NaN</td>
      <td>WASHINGTON</td>
      <td>NaN</td>
      <td>2</td>
      <td>HURLEY</td>
      <td>950</td>
    </tr>
    <tr>
      <th>7</th>
      <td>100008000.0</td>
      <td>213.0</td>
      <td>TREMONT</td>
      <td>Y</td>
      <td>--</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>100009000.0</td>
      <td>215.0</td>
      <td>TREMONT</td>
      <td>Y</td>
      <td>na</td>
      <td>2</td>
      <td>1800</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['PID', 'ST_NUM', 'ST_NAME', 'OWN_OCCUPIED', 'NUM_BEDROOMS', 'NUM_BATH',
           'SQ_FT'],
          dtype='object')



Now we can start to answer some of our questions, **what are my features?** It is easy to infer this from the column names.

- **<code>PID</code>**: Property ID
- **<code>ST_NUM</code>**: Street number
- **<code>ST_NAME</code>**: Street name
- **<code>OWN_OCCUPIED</code>**: Is the residence owner occupied
- **<code>NUM_BEDROOMS</code>**: Number of bedrooms
- **<code>NUM_BATH</code>**: Number of bathrooms
- **<code>SQ_FT</code>**: Square footage of the property

We can also answer, **what are the expected types?**
- **<code>PID</code>**: float or int (some sort of numeric type)
- **<code>ST_NUM</code>**: float or int (some sort of numeric type)
- **<code>ST_NAME</code>**: string
- **<code>OWN_OCCUPIED</code>**: string where Y ("Yes") or N ("No")
- **<code>NUM_BEDROOMS</code>**: float or int(some sort of numeric type)
- **<code>NUM_BATH</code>**: float or int (some sort of numeric type)
- **<code>SQ_FT</code>**: float or int (some sort of numeric type)

When we look at what the actual data types are, some of them don't match. 


```python
df.dtypes
```




    PID             float64
    ST_NUM          float64
    ST_NAME          object
    OWN_OCCUPIED     object
    NUM_BEDROOMS     object
    NUM_BATH         object
    SQ_FT            object
    dtype: object



Part of the reason that NaN is a 'float' data type which then gets mixed in with 'string' data types. 

An example of this is the "Owner Occupied" column.


```python
# Check type of each record
df['OWN_OCCUPIED'].map(type)
```




    0      <class 'str'>
    1      <class 'str'>
    2      <class 'str'>
    3      <class 'str'>
    4      <class 'str'>
    5      <class 'str'>
    6    <class 'float'>
    7      <class 'str'>
    8      <class 'str'>
    Name: OWN_OCCUPIED, dtype: object



It's important to note that row 4 has a value of 12 but is categorized as a string data type. This isn't a big deal since we'll be altering that value anyways but it's good to make note that some of the numerics in the dataset may need to be changed in the end to get the desired data type.

Let's start digging into answering the next two questions on detecting missing values.
## Standard Missing Values
What missing values can we detect with Pandas? Let's take a look at our raw dataset again.

|    PID    | ST_NUM | ST_NAME    | OWN_OCCUPIED | NUM_BEDROOMS | NUM_BATH | SQ_FT |
|-----------|--------|------------|--------------|--------------|----------|-------|
| 100001000 | 104    | PUTNAM     | Y            | 3            | 1        | 1000  |
| 100002000 | 197    | LEXINGTON  | N            | 3            | 1.5      | --    |
| 100003000 |        | LEXINGTON  | N            | n/a          | 1        | 850   |
| 100004000 | 201    | BERKELEY   | 12           | 1            | NaN      | 700   |
|           | 203    | BERKELEY   | Y            | 3            | 2        | 1600  |
| 100006000 | 207    | berkeley   | Y            | NA           | 1        | 800   |
| 100007000 | NA     | WASHINGTON |              | 2            | HURLEY   | 950   |
| 100008000 | 213    | TREMONT    | Y            | --           | 1        |       |
| 100009000 | 215    | TREMONT    | Y            | na           | 2        | 1800  |

In the "Stree Number" column there is an empty cell in the 3rd row and an "NA" value in the 7th row. 

These are clearly both missing values. Let's see how Pandas deals with these.


```python
# Looking at the ST_NUM column
print(df['ST_NUM'])
print("\n") # line break for readability
print(df['ST_NUM'].isnull())
```

    0    104.0
    1    197.0
    2      NaN
    3    201.0
    4    203.0
    5    207.0
    6      NaN
    7    213.0
    8    215.0
    Name: ST_NUM, dtype: float64
    
    
    0    False
    1    False
    2     True
    3    False
    4    False
    5    False
    6     True
    7    False
    8    False
    Name: ST_NUM, dtype: bool
    

We can see that Pandas has filled in both the blank and "NA" records with NaN. Using the <code>isnull()</code> method confirms that both values were recognized as missing values as the boolean responses are <code>True</code>.

These two missing records were handled by Pandas with no issues. Now let's take a look at some types that Pandas won't recognize.
## Non-standard Missing Values
Sometimes there are cases where the missing values have different formats. 

Let's take a look at raw values of the NUM_BEDROOMS column as an example.

| NUM_BEDROOMS |
|--------------|
| 3            |
| 3            |
| n/a          |
| 1            |
| 3            |
| NA           |
| 2            |
| --           |
| na           |

Here there are four missing values.
- n/a
- NA
- -\-
- na

From the previous section we know that Pandas will be able to handle NA but what about the other values?


```python
# Looking at the NUM_BEDROOMS column
print(df['NUM_BEDROOMS'])
print("\n") # line break for readability
print(df['NUM_BEDROOMS'].isnull())
```

    0      3
    1      3
    2    NaN
    3      1
    4      3
    5    NaN
    6      2
    7     --
    8     na
    Name: NUM_BEDROOMS, dtype: object
    
    
    0    False
    1    False
    2     True
    3    False
    4    False
    5     True
    6    False
    7    False
    8    False
    Name: NUM_BEDROOMS, dtype: bool
    

Pandas was able to recognize "NA" and "n/a" as missing values. However, the "--" and "na" types were not.

This could occur when multiple users manually enter data and each one has their own preference of entering a field as null.

One way to detect these various formats is to put them into a list. Then when we import the data, Pandas will recognize them as missing values.


```python
# Making a list of missing value types
missing_values = ["na", "--"]
df = pd.read_csv("property data.csv", na_values=missing_values)
```

Now let's take a look to see what the NUM_BEDROOMS column looks like now.


```python
# Take a look at the NUM_BEDROOMS column
print(df['NUM_BEDROOMS'])
print('\n')
print(df['NUM_BEDROOMS'].isnull())
```

    0    3.0
    1    3.0
    2    NaN
    3    1.0
    4    3.0
    5    NaN
    6    2.0
    7    NaN
    8    NaN
    Name: NUM_BEDROOMS, dtype: float64
    
    
    0    False
    1    False
    2     True
    3    False
    4    False
    5     True
    6    False
    7     True
    8     True
    Name: NUM_BEDROOMS, dtype: bool
    

By creating and referencing a list of different data types that represent missing values we were able to get Pandas to recognize them as such. This also addresses the "--" value seen in the 2nd row of the "square feet" column.

In larger datasets it may be harder to find these but as you go through the data you can add them to the missing_values list.

## Unexpected Missing Values
So far we've gone through standard and non-standard types of missing values. What do we do when we encounter an unexpected type?

For example, if we expect a feature to be a string but there is a numeric type then it could be considered a missing value since it doesn't provide any value to us.

An example of this is in the "owner occupied" column where there is a value of 12 in row 4.

| OWN_OCCUPIED |
|--------------|
| Y            |
| N            |
| N            |
| 12           |
| Y            |
| Y            |
|              |
| Y            |
| Y            |

We already know the blank space in row 7 will be detected as a missing value by Pandas. Let's confirm this real quick.


```python
print(df['OWN_OCCUPIED'])
print('\n')
print(df['OWN_OCCUPIED'].isnull())
```

    0      Y
    1      N
    2      N
    3     12
    4      Y
    5      Y
    6    NaN
    7      Y
    8      Y
    Name: OWN_OCCUPIED, dtype: object
    
    
    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    7    False
    8    False
    Name: OWN_OCCUPIED, dtype: bool
    

The value in row 4 is the number 12. We expect and require the response for Owner Occupied to be a string of either Y or N. 

Since we don't have any information to help us determine if the number 12 should correspond to a Y or N we will label this as a missing value.

We'll use a for loop to change any integer values to be a missing value.


```python
df['OWN_OCCUPIED']
```




    0      Y
    1      N
    2      N
    3     12
    4      Y
    5      Y
    6    NaN
    7      Y
    8      Y
    Name: OWN_OCCUPIED, dtype: object




```python
# Detecting numbers
counter = 0
for row in df['OWN_OCCUPIED']:
    try:
        int(row)
        df.loc[counter, 'OWN_OCCUPIED']=np.nan
    except ValueError:
        pass
    counter +=1
```

Let's double check that it worked:


```python
print(df['OWN_OCCUPIED'])
print("\n")
print(df['OWN_OCCUPIED'].isnull())
```

    0      Y
    1      N
    2      N
    3    NaN
    4      Y
    5      Y
    6    NaN
    7      Y
    8      Y
    Name: OWN_OCCUPIED, dtype: object
    
    
    0    False
    1    False
    2    False
    3     True
    4    False
    5    False
    6     True
    7    False
    8    False
    Name: OWN_OCCUPIED, dtype: bool
    

Success! Now we can use the same for loop with a little alteration and apply it to the "Number of baths" column to replace the value "HURLEY" as missing.

| NUM_BATH |
|----------|
| 1        |
| 1.5      |
| 1        |
| NaN      |
| 2        |
| 1        |
| HURLEY   |
| 1        |
| 2        |


```python
# Detecting strings
counter = 0
for row in df['NUM_BATH']:
    try:
        float(row)
        df.loc[counter, 'NUM_BATH']=float(row)
    except ValueError:
        pass
    counter +=1

df['NUM_BATH'] = df['NUM_BATH'].apply(lambda x: np.nan if type(x) == str else x)
```


```python
# Double check
print(df['NUM_BATH'])
print("\n")
print(df['NUM_BATH'].isnull())
```

    0    1.0
    1    1.5
    2    1.0
    3    NaN
    4    2.0
    5    1.0
    6    NaN
    7    1.0
    8    2.0
    Name: NUM_BATH, dtype: float64
    
    
    0    False
    1    False
    2    False
    3     True
    4    False
    5    False
    6     True
    7    False
    8    False
    Name: NUM_BATH, dtype: bool
    

## Summarizing Missing Values
After cleaning up our missing values, we'll summarize them in different ways.

First let's take a look at our dataset again.


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PID</th>
      <th>ST_NUM</th>
      <th>ST_NAME</th>
      <th>OWN_OCCUPIED</th>
      <th>NUM_BEDROOMS</th>
      <th>NUM_BATH</th>
      <th>SQ_FT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001000.0</td>
      <td>104.0</td>
      <td>pUTNAM</td>
      <td>Y</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002000.0</td>
      <td>197.0</td>
      <td>LEXINGTON</td>
      <td>N</td>
      <td>3.0</td>
      <td>1.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003000.0</td>
      <td>NaN</td>
      <td>LEXINGTON</td>
      <td>N</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>850.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004000.0</td>
      <td>201.0</td>
      <td>BERKELEY</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>700.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>203.0</td>
      <td>BERKELEY</td>
      <td>Y</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100006000.0</td>
      <td>207.0</td>
      <td>berkeley</td>
      <td>Y</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>800.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100007000.0</td>
      <td>NaN</td>
      <td>WASHINGTON</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>950.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>100008000.0</td>
      <td>213.0</td>
      <td>TREMONT</td>
      <td>Y</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>100009000.0</td>
      <td>215.0</td>
      <td>TREMONT</td>
      <td>Y</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1800.0</td>
    </tr>
  </tbody>
</table>
</div>



Now let's take a look at the total number of missing values for each feature.


```python
# Total missing values for each feature
df.isnull().sum()
```




    PID             1
    ST_NUM          2
    ST_NAME         0
    OWN_OCCUPIED    2
    NUM_BEDROOMS    4
    NUM_BATH        2
    SQ_FT           2
    dtype: int64



There are times when we want to do a quick check to see if there are any missing values in the entire dataset.


```python
# Any missing values?
df.isnull().values.any()
```




    True



We can also get the total count of missing values.


```python
# Total number of missing values
df.isnull().sum().sum()
```




    13



Now that we identified and correctly labeled the missing values it's time to deal with them.

## Replacing
The easiest way to deal with missing values is to simply drop the record however you could be throwing away valuable information if the other columns are populated for that record.

We'll be doing some basic imputations here but a detailed statistical approach to dealing with missing data coupled with good engineering judgement should be used for real-life models.

We could choose to fill in all missing values with a single value.


```python
# Replace missing values with a number
print(df['ST_NUM'])
df['ST_NUM'].fillna(125, inplace=True)
print('\n')
print(df['ST_NUM'])
```

    0    104.0
    1    197.0
    2      NaN
    3    201.0
    4    203.0
    5    207.0
    6      NaN
    7    213.0
    8    215.0
    Name: ST_NUM, dtype: float64
    
    
    0    104.0
    1    197.0
    2    125.0
    3    201.0
    4    203.0
    5    207.0
    6    125.0
    7    213.0
    8    215.0
    Name: ST_NUM, dtype: float64
    

More likely, we would be using a location based imputation.


```python
# Location based replacement
print(df['NUM_BEDROOMS'])
df.loc[2,'NUM_BEDROOMS'] = 2
print('\n')
print(df['NUM_BEDROOMS'])
```

    0    3.0
    1    3.0
    2    NaN
    3    1.0
    4    3.0
    5    NaN
    6    2.0
    7    NaN
    8    NaN
    Name: NUM_BEDROOMS, dtype: float64
    
    
    0    3.0
    1    3.0
    2    2.0
    3    1.0
    4    3.0
    5    NaN
    6    2.0
    7    NaN
    8    NaN
    Name: NUM_BEDROOMS, dtype: float64
    

A widely used method of replacing missing values is to use a median.


```python
# Replace using median
print(df['NUM_BATH'])
median = df['NUM_BATH'].median()
df['NUM_BATH'].fillna(median, inplace=True)
print('\n')
print(df['NUM_BATH'])
```

    0    1.0
    1    1.5
    2    1.0
    3    NaN
    4    2.0
    5    1.0
    6    NaN
    7    1.0
    8    2.0
    Name: NUM_BATH, dtype: float64
    
    
    0    1.0
    1    1.5
    2    1.0
    3    1.0
    4    2.0
    5    1.0
    6    1.0
    7    1.0
    8    2.0
    Name: NUM_BATH, dtype: float64
    

These are very basic methods of handling missing data. There are much more sophisticated ways of handling missing data that I don't go into here since a lot of it depends on a number of factors. Each dataset comes with its own nuances that the Data Scientist has to learn to navigate and make determinations on.

## Miscellaneous Cleaning
Often times when dealing with name type columns you'll encounter words that are capitalized, lowercase, or a combination of both. These name type columns aren't always used however if you're looking to extract some info from it then some cleaning is required.

For the "Street Name" column all records are uppercase except in row 6 where the value is all lowercase "berkeley" and row 1 where the first letter is lowercase and the rest is uppercase "pUTNAM".

We'll choose to standardized all values to use uppercase.


```python
df['ST_NAME']
```




    0        pUTNAM
    1     LEXINGTON
    2     LEXINGTON
    3      BERKELEY
    4      BERKELEY
    5      berkeley
    6    WASHINGTON
    7       TREMONT
    8       TREMONT
    Name: ST_NAME, dtype: object




```python
df['ST_NAME'] = df['ST_NAME'].apply(lambda x: x.upper())
df['ST_NAME']
```




    0        PUTNAM
    1     LEXINGTON
    2     LEXINGTON
    3      BERKELEY
    4      BERKELEY
    5      BERKELEY
    6    WASHINGTON
    7       TREMONT
    8       TREMONT
    Name: ST_NAME, dtype: object



## Conclusion
Messy data is inevitable. Data cleaning is part of the process in any data science project.

Although the dataset used here was very small, these techniques can easily be applied to much larger datasets.
