# Homework 2: Data Structures in Python

### Due Friday, 09.06.2024 at 11:59 PM ET

## Goals

This homework has several objectives:

1. Write some basic Python programs.
2. Get familiar with the different data structures available in Python.
3. Leverage the concept of functions to write modular code.

## Instructions

In this homework, you need to write two Python functions, one per problem described below. Both of these function definitions are provided to you in `homework2.py`. The `testhistox.py` and `testbirthdayx.py` can be used by you to test your functions in `homework2.py`. We have provided you with some test cases, you may make your own test case and and execute them to make sure your code runs properly.

### Problem 1

Create a function called `histogram` that takes as input a **dictionary** which contains following fields: dataset `data`, a lower bound `min_val`, an upper bound `max_val`, and a number of bins `n`, and returns a histogram representation of `data` with `n` bins between these bounds. Specifically, your function should:

1. Have input arguments `histogram(input_dictionary)`, which, within the dictionary, it is expecting `data` as a list of floats, `n` as an integer, and `min_val` and `max_val` as floats.

2. Print the error message `Error: min_val and max_val are the same value` and return an empty list if `min_val` and `max_val` are the same number (the width of the histogram is 0).

3. At the beginning of your function, be sure to check that `n` is a positive integer; if not, your code should just return an empty list for `hist`.

4. If `min_val` is larger than `max_val`, re-assign `min_val` to `max_val` and `max_val` to `min_val`.

5. If n is equal to 0, return an empty list.

6. Initialize the histogram `hist` as a list of `n` zeros.

7. Calculate the bin width as `w = (max_val-min_val)/n`, so that `hist[0]` will represent values in the range `(min_val, min_val + w)`, `hist[1]` in the range `[min_val + w, min_val + 2w)`, and so on through `hist[n-1]`. (Remember that [ is inclusive while ) is not!)

8. Ignore any values in `data` that are less than or equal to `min_val` and greater than or equal to `max_val`. **Remember, if you have changed `max_val` and `min_val` in step 4, you will need to work with the new values of `max_val` and `min_val`.**

9. Increment `hist[i]` by 1 for each value in `data` that belongs to bin `i`, i.e., in the range `[min_val + i*w, min_val + (i+1)*w)`.

10. Return `hist`.



For example, typing in

```
data = [-2, -2.2, 0, 5.6, 8.3, 10.1, 30, 4.4, 1.9, -3.3, 9, 8]
input_dictionary = {'data': data, 'n': 15, 'min_val': -5, 'max_val': 10}
hist = histogram(input_dictionary)
print(hist)
```

should return

```
[0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1]
```
Some other test cases are:

```
data = [-4, -3.2, 0, 7.6, 1.0, 2.2, 30, 2.2, 1.9, -8.3, 6, 5]
input_dictionary = {'data': data, 'n': 10, 'min_val': 10, 'max_val': 0}
hist = histogram(input_dictionary)
print(hist)
```

should return

```
[0, 2, 2, 0, 0, 1, 1, 1, 0, 0]
```
and,
```
data = [2,2,2]
input_dictionary = {'data': data, 'n': 5, 'min_val': -2, 'max_val': 3}
hist = histogram(input_dictionary)
print(hist)
```
returns
```
[0, 0, 0, 0, 3]
```
also, 
```
data = [-1,-1,-1,10,10]
input_dictionary = {'data': data, 'n': 5, 'min_val': -1, 'max_val': 10}
hist = histogram(input_dictionary)
print(hist)
```
returns 
```
[0, 0, 0, 0, 0]
```
Note: Please include all conditions specified in this problem in your code. 

### Problem 2

Create a function called `combine_birthday_data` that takes in three lists of tuples, `person_to_day`, `person_to_month`, and `person_to_year`, and combines them into a single dictionary `month_to_people_data`. Specifically, your function should:

Have input arguments `combine_birthday_data(person_to_day, person_to_month, person_to_year)`, expecting `person_to_day` as a list of tuples structured as (name (string), day (integer)), where the name represents the person's name and day represents the day of the birthday, `person_to_month` as a list of tuples structured as (name (string), month (integer)), where the month is the month of the birthday, and `person_to_year` as a list of tuples structured as (name (string), year (integer)) mapping a person's name to their birthday year in the tuple. You may assume all inputs to be valid.

Create a new dictionary `month_to_people_data` where the keys are all the months contained in `person_to_month` (note: if a month does not appear in `person_to_month`, it should not be included in `month_to_people_data`), and the values are people born in that month. If there is more than one person born in a particular month, *only then* convert the value to a list. If there is only one person born in a month, then setting it to a list will result in no points being awarded for the particular test case. Each person is represented by a tuple of the structure `(name, day, year, age)`. *The age is calculated by subtracting the current year (2024) from the year in `person_to_year`.*
Return `month_to_people_data`.

For example, if the input is:

```
person_to_day = [("John", 5), ("Jane", 10), ("Mike", 20), ("Lucy", 23), ("Sam", 6)]
person_to_month = [("John", 3), ("Jane", 5), ("Mike", 5), ("Lucy", 3), ("Sam", 10)]
person_to_year = [("John", 1990), ("Jane", 1995), ("Mike", 2000), ("Lucy", 2002), ("Sam", 2023)]

```

The function should return:

```
{3: [('John', 5, 1990, 34), ('Lucy', 23, 2002, 22)], 5: [('Jane', 10, 1995, 29), ('Mike', 20, 2000, 24)], 10: ('Sam', 6, 2023,1)}

```
## Testing

We have provided testcases for you, that recreate the examples from above, in `testhisto1.py`, `testhisto2.py`, `testhisto3.py`, `testhisto4.py`, `testbirthday1.py`, `testbirthday2.py` and `testbirthday3.py`, which test problems 1 and 2, respectively. Note that these test programs will only work "out of the box" if you have your solution in `homework2.py`. You may verify your code by running the test programs from the terminal. The concept of importing functions from modules or `.py` files is being used here.


## What to Submit

Put the two functions `histogram` and `combine_birthday_data` in a single file called `homework2`.

Once you have a version of this file (that you have `commit`ted using `git commit` and `push`ed using `git push`) that you are happy with, you are done!
Sit back, relax and enjoy your lectures :)
