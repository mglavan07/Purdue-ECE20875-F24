# Homework 6: Regular Expressions
## Due: 11 October 2024 at 11:59 pm ET

The length of this homework is inversely proportional to your knowledge in writing regular expressions, both for finding matches and for doing substitutions.

A good resource to use is https://regex101.com/ for checking your expressions.

## Background

Please refresh your memory of regular expressions using the class notes. You may also find the Python [documentation on regular expressions](https://docs.python.org/3/library/re.html) useful.

A few helpful reminders: 

### Testing for Patterns

When you use `re.search` to find a regular expression match, it returns a `Match` object if the pattern exists in the string (we will see more about objects later in the semester). If *there is no match*, then `re.search` (and `re.match` and `re.findall`) will return `None`,  which you can test for as shown below:

```
p = re.compile('pattern')
if (p.search(s)):
   # This branch will execute if the pattern is found
else:
   # This branch will execute if the pattern is *not* found
```

### Substituting with functions

A common use of `re.sub` is to substitute one string for another (remember that you can use the *groups* that you match in a pattern as part of your string substitution):

```
s = "loooool"
p = re.compile('(l)o+(l)')
p.sub(r'\1o\2', s) # replace "loooool" with "lol"
```

You can also call a method instead of providing a replacement string. This method will be called with the `Match` object corresponding to the matched string, and should return a string:

```
def replFun(m) :
   return m.group(2).upper()
s = "loooool"   
p = re.compile('(l)(o+)(l)')
m = p.search(s)
p.sub(r'\1'+replFun(m) +r'\3', s) #replace "loooool" with "lOOOOOl"
```
The `re.MatchObject.group()` method returns the complete matched subgroup by default or a tuple of matched subgroups depending on the number of arguments. Remember that `(`, `)`, `-` and `.` are special characters for regular expressions. To search for those characters, you need to precede them with a backslash: `\(` `\)`, `\-`, `\.`. 

# Instructions

## 0) Set up your repository

The repository should contain two files:

1. `problems.py`, the file in which you will fill in the functions for the problems. This also contains test code you can use to test your solutions.
2. This README.

## Problem 1: Regular expression matches

Congratulations! You have been hired as a new agent of SHIELD (security organization that defends Earth). The world is getting more and more superheroes and your first task is to work on the database of superhero emails (even those with telepathic powers need a traditional way to communicate). However, it appears that the evil forces are at work, the nefarious Doctor Octavious has hacked SHIELDs database and has entered in false emails. You must create a function that can correctly identify which emails are valid.

A valid email at SHIELD has the following structure:

Fill in the function `problem1`. This function should return the string `valid` if the input string *is a valid email address* and `invalid` if not. We define a valid email as follows:

1. The email **must** begin with the ID number of the hero, from 100 up to 799 (This is dictated by SHIELD's numbering rules) followed by a "."(period)
2. After the "." the email **must** have a name composed of upper or lower case letters, containing at least 1, but no more than 10 letters.
3. The email **may** have any number of numbers following the letters of the name, but anything else between the name and the @ symbol is invalid
4. The email **must** have the “@” symbol followed by either "shield.gov" or "starkindustries.com"
5. There **must** be nothing following the .gov or .com

Correct Examples:
```
123.iamironman@starkindustries.com
250.Srogers1776@starkindustries.com
100.nickfury@shield.gov
```

Incorrect Examples:
```
144.venom@starkindustries.comasdf (broke rule 5)
942.hyperion@starkindustries.com (breaks rule 1, incorrect number in front)
567.greengoblin@shield.gov (breaks rule 2, too many letters)
324drdoom324@starkindustries.com (breaks format established in 1 (does not have "."))
765.Hosborn*876@shield.gov (breaks rule 3, has something other than numbers after name and between "@" symbol)
234.vulture@shield.com (breaks rule 4, not correct ending)
```

*ANY other format should not count as a valid email. Spaces before or after an otherwise valid email is considered invalid.*

Because we are looking for the entire string to be an email, you can either use `^` and `$` to force a match to be at the beginning and end of a string, or you can use `fullmatch` instead of `match` or `search`.

## Problem 2: Groups

You want to explore other jobs and now you’ve been hired as a junior librarian (it doesn’t pay well, but hey, you get to sit around books all day). The library has a unique problem: all the books are shelved by sentences describing the author and the book. Unfortunately, they’re not in any particular order. Your task is to extract the author's name and the title of the book from these sentences, so the books can finally be put back in the right place.

A sentence would have the format *Author wrote Book Name* or *Author wrote books* and will have the following conditions:

1. An author's name can be one word, or two words. The word or words in a name must start with a capital letter.
2. The verb is "wrote".
3. If the book has a name, it can be one, two or three words/numbers. Each word starts with a capital letter or number. If the book does not have a name, "wrote" will be followed by "books". 
4. If the above conditions are both not met, there is no match. You will then return a tuple of strings ("noauthor", "noname")

Example: 

`JK Rowling wrote the Harry Potter Series`

Although there is the phrase “JK Rowling wrote” it has "the" following the word "wrote" and its first letter is not capital. Therefore, this will not meet the conditions.

Fill in the function `problem2`. This function should search an input string for the author and book then return a tuple of them.

`George Orwell wrote 1984`

you should return:

`('George Orwell','1984')`

If you pass in:

`In the 1930s, a Mystery writer wrote Mary Westmacotts. Later it was found that Agatha Christie wrote The Westmacott Novels`

you should return:

`('Agatha Christie', 'The Westmacott Novels')`

*This is because Mystery writer did not have a capitalized first letter in the word preceding "wrote"*

If you pass in: 

`Roxette wrote books`

you should return:

`('Roxette', 'books')`

_book does not have a name - returned value is just one word and books_


**Be careful not to return (leading or trailing) extra spaces in the return value. You may need to do a little bit of extra processing of the string captured by your group to ensure this. You will receive partial credit for having spaces. Please remove extra spaces for full credit.**

## Problem 3: Substitution

You found the library job too boring and begged SHIELD to hire you again. You have a new super hero problem to solve now! Someone has changed these audio transcripts from mission reports. You must correct the super hero name!  

Replace the word Boy/Girl or boy/girl with Man/Woman. Fill in the function `problem3` to returns a string with the correct word restored. 

Here are the rules for replacement
1. Boy/Girl or boy/girl should be replaced with Man/Woman if the word prior to it is a name (starts with a capital letter)
2. If no match is found, return "nomatch"

Here are some examples: 

`Spider Girl, I need help!`

*Should return*

`Spider Woman, I need help!`


`There is a boy trapped in a burning building Iron Boy`

*Should return*

`There is a boy trapped in a burning building Iron Man`

_This is because the first boy is not proceded by a word that has a Capital letter_


**Be careful not to return extra spaces in the final output. You may need to do a little bit of extra processing of the string captured by your group to ensure this. You will receive partial credit for having unwanted spaces. Please remove extra spaces for full credit.**

# Testing Your Code

To test your code, run the `problems.py` file. These tests are not exhaustive, passing them does not guarantee full credit on the homework.

# What to Submit

Please submit `problems.py` with all the functions filled in. 

# Submitting your code

Please add, commit and push the latest version of your code, as you did in the previous Homeworks.
Do not make any modifications after submission to avoid a late penalty.
