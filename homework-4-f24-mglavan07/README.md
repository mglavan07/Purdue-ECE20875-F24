# Homework 4: Higher-order Functions
## Due: 09/20/2024 11:59 PM ET

This homework primarily covers defining and using higher-order functions.

# Goals

In this homework you will:

1. Write higher order functions that accept one or more functions as an input.
2. Write higher order functions that return a function as output.
3. Develop a basic signal processing algorithm (convolution).

# Background and Instructions

## 1) Problem 1: Higher-order functions

Problem 1 will get you familiar with several ways to build and use higher order functions.
In this problem, we ask you to fill in the code for a number of missing functions in `problem1.py`:

1. `fun_map`: This takes in two function, `func_1` and `func_2`, and a list `L` as its arguments and returns a list, which consists of applying `func_1` to every odd index position element of `L` and `func_2` to every even index position element of `L`. i.e., return a list `R` which

	1.1. performs func_1 on every odd index of L => `func_1(i)` where `i=1,3,5,... `

	1.2. performs func_2 on every even index of L => `func_2(i)` where `i=0,2,4,...`

3. `compose_map`: This takes in two functions, `func_1` and `func_2`, and a list `L` as its arguments and returns a list, which consists of applying `func_1` then `func_2` to each element of the input list `L`. Note that you must apply the functions in that order: call `func_1` first, then `func_2` on the output of `func_1` for each element in the input list `L`.

4. `compose`: This takes in two functions, `func_1` and `func_2`, and returns a new function `ret_fun`. The new function takes in a single argument `i` then calls `func_2` followed by `func_1` on that argument, returning the result. Note again that the order matters. *Note: Expected return type for `compose` is a function and not a value.*

5. `repeater`: This takes in a function `fun` and an integer `num_repeats`, and returns a new function, `ret_fun`. The new function takes in an input `x` and calls `fun` on it `num_repeats` times (in other words, you call `fun` on the output of calling `fun` on the output of calling `fun` ... on the input, repeated as many times as `num_repeats`). *Expected return type for `repeater` is a function and not a value.*

### Testing

If your code works, and you run the test code provided in `problem1.py`, you should get the following output:

```
[-2, 6, 2, 14, -12, -18, 8, 26, 12, 34, 16]
[-1, 3, 7, 11, -21, -21, 19, 23, 27, 31, 35]
[-4, 0, 4, 8, -24, -24, 16, 20, 24, 28, 32]
2
[-4, 0, 4, 8, -24, -24, 16, 20, 24, 28, 32]
5
[-1, 3, 7, 11, -21, -21, 19, 23, 27, 31, 35]
repeat 0 times: 5
repeat 1 time: 10
repeat 2 times: 20
repeat 3 times: 40
```

## 2) Problem 2: Stencils

In Problem 2, we ask you to write two functions. The first is a generic
*stencil* function, and the second is a function that generates *box filters*
that, when combined, will let you perform a specific operation known as 
[convolution](https://en.wikipedia.org/wiki/Convolution) (Explained later in the section).

### Stencils
A stencil (in the most general sense) is a way of computing a value for a given
data point by using the values of neighboring data points. Suppose we have a
*stencil function* `f`. `f` is a function that accepts a list of some fixed
length (for the purposes of explanation, let's say 3) and outputs a single value that is some
combination of the values in that list. Here is one example:

```
def f(L) :
    return L[0] * L[1] + L[2]
```

Applying a stencil function means treating the stencil function like a "window"
and "sliding" it across a list, computing the result of the stencil function at
each point. So if we start with the following list:

```
[a, b, c, d, e, f, g]
```

Applying the stencil function produces the following output list:

```
[f([a, b, c]), f([b, c, d]), f([c, d, e]), f([d, e, f]), f([e, f, g])]
```

> Note two things:
> 1. The length of the output list is smaller than the length of the input list. If the input list is of length `k`, the output list will be of length `k - width + 1`, where `width` is the width of the stencil that `f` applies.
> 2. You can think of a stencil as a generalization of map that uses neighboring elements to compute the output instead of just the element itself. If `f` has a window of width 1, the output is basically like using map function.

We will ask you to write a function for applying stencils that takes in an input list, a stencil function, and a third parameter that tells you how wide the stencil function is, in the case shown above, the stencil function uses a width of 3 to slide over the list `[a, b, c, d, e, f, g]`.

### Filter, Cross-correlation and Convolution

Given a filter of length `k` that specifies a list of `k` numbers (for a 1-D box filter), when applied to a input signal, computes its result by multiplying the first element of the filter by the first element of the input, the second element of the filter by the second element of the input, and so on, and then adding all the results together to output one single value. 

In the signal processing literature, a filter is a signal that when applied to a target signal, removes some unwanted components or features from the target signal.

So if you have a filter (list), say `b`, of size 3 with elements `b[0], b[1], b[2]` and you slide it over the list `[a, b, c, d, e, f, g]`, the expected output would be:

```
[b[0] * a + b[1] * b + b[2] * c,
 b[0] * b + b[1] * c + b[2] * d,
 b[0] * c + b[1] * d + b[2] * e,
 b[0] * d + b[1] * e + b[2] * f,
 b[0] * e + b[1] * f + b[2] * g]
```

In one dimension (1-D), applying a filter in this way (shown above) computes the (discrete) 
cross-correlation of the signal in the input list with the filter. This is a measure of similarity between the signal and the filter. 

Additionally, the convolution operation in 1-D is very similar to the cross-correlation operation but has a slight variation. In the convolution operation, the filter is first flipped and then applied to the input signal. In other words, convolution is simply a cross-correlation on the input signal with a flipped version of the given filter.

So, if you have a filter with elements `b[0], b[1], b[2]` and you perform a convolution operation on the list `[a, b, c, d, e, f, g]`, you are basically sliding over the list with a flipped version of the filter, hence, the output you get would be:

```
[b[2] * a + b[1] * b + b[0] * c,
 b[2] * b + b[1] * c + b[0] * d,
 b[2] * c + b[1] * d + b[0] * e,
 b[2] * d + b[1] * e + b[0] * f,
 b[2] * e + b[1] * f + b[0] * g]
``` 
Convolution conveys how the shape of signal is modified by the filter.


> In two dimensions, filters can be used to perform image-processing tasks 
> like blurring and edge detection. These types of filters are also one of 
> the key steps in deep neural networks that operate on images (they form the 
> core of the "convolution" layers).

In the second part of Problem 2, we ask you to write a function that, when given a list of filter elements, *creates a stencil function* from those elements.


Finally, coming to the point, we ask you to fill in the two functions in `problem2.py`:

1. `stencil(data, f, width)`: This function takes in a list `data`, a function `f`, and an int `width`, and returns a list. This function returns an output list of length `k-width+1`, the result of applying the stencil function `f` to the input list `data` 

2. `create_box(box)`: This function accepts a list `box`, and returns two outputs: a new function and a width (int). The width is the length of the box (the number of elements that the filter looks at). The new function is a stencil function that operates on items from an input list and applies a convolution operation to them as described above. The function should check, at the beginning, if the length in the input parameter is the same as the length of `box`. If not, the following error should be printed: 

	`Calling box filter with the wrong length list. Expected length of list should be n.`

    where `n` is the length of `box`. For example if `n=7`, the printed warning should be:

	`Calling box filter with the wrong length list. Expected length of list should be 7.`

    and then after the print statement, return 0.

Moreover, the comments under `create_box()` in problem2.py provide further explanation on what `create_box(box)` is supposed to do.


### `pass` statement in the function definitions
The `pass` statement is used as a placeholder for future code (as done by us).
When the `pass` statement is executed, nothing happens, but you avoid getting an error when you have empty code.
Empty code is not allowed in loops, function definitions, class definitions, or in if statements.
You should delete/comment out the `pass` statement of a function once you write your code for that particular function.

### Testing

If your code works, and you run the test code provided in `problem2.py`, you should get the following output:

```
[3.0, 5.0, 1.0, -3.6666666666666665, -2.3333333333333335, 5.0, 13.0, 15.0, 17.0]
[165, 245, 357, 501, 677, 885, 1165]
[3.0, 5.0, 0.9999999999999996, -3.666666666666667, -2.3333333333333335, 5.0, 13.0, 14.999999999999998, 17.0]
[-3.0, 6.0, 7.0, -2.0, -11.0, -12.0, -3.0, -3.0]
```

(The floating point numbers may be rounded a little bit differently, depending on exactly how you implement your box filter.)

### Helpful documentation 
Python slicing https://www.geeksforgeeks.org/python-list-slicing/

List comprehension https://www.w3schools.com/python/python_lists_comprehension.asp

# What to Submit?

For Problem 1, please submit `problem1.py` with all the appropriate functions filled in.

For Problem 2, please submit `problem2.py` with all the appropriate functions filled in.

# Submitting your code

Your final submission should have modified `problem1.py` and `problem2.py` files. Ensure to push all changes to your repository by 11:59PM on 09/20/2024.
