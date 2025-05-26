# Homework 5: Sampling, Hypothesis Testing, and Confidence Intervals
## Due: 09/27/2024 (NOTE: You have *one* week to complete this homework. This is a challenging homework, please start early!)
This assignment is designed to enhance your skills in hypothesis testing and confidence intervals through practical application using Python. Additionally, it marks the first assignment that incorporates a written report component, to be submitted in PDF format.

# THE PDF DOCUMENT WILL HAVE TO BE SUBMITTED ON GRADESCOPE and Code pushed to Github as usual!

# Goals

In this assignment, you will:

1. Develop hypotheses and conduct suitable statistical analyses to test them.
2. Calculate confidence intervals, grounded in relevant assumptions.
3. Engage with an authentic dataset, collected via a survey.

# What to Submit

#### For this assignment, please submit your responses in a "Writeup.pdf" file. This document is a fillable PDF, allowing you to either type your answers directly into the file and save it or to input your responses using digital writing tools. To ensure you receive full credit for your work, please upload the completed PDF to GitHub and submit it to GradeScope for grading. Remember, including your PDF in your GitHub repository submission is also crucial.

Please ensure the following components are submitted for evaluation:

(i) Upload the "Writeup.pdf" file, with all answers completed, to GradeScope and GitHub.

(ii) Submit the "problem2.py" file to GitHub, showcasing your solution for Problem 2.

(iii) Similarly, submit the "problem3.py" file to GitHub, showcasing your solution for Problem 3.

Ensure that your writeup clearly presents the answers to each question, including detailed explanations of your results where requested. While it is advisable to have your code files output the calculated values for each question, doing so is optional.

# Background

Before you start the homework, we recommend revisiting the materials on sampling and hypothesis testing available on the course website. You are encouraged to use and adapt any code provided there to assist with your work.

## Understanding Hypothesis Testing: A Practical Guide with an Example

Hypothesis testing is a statistical method used to infer the validity of a claim about a population based on sample data. It starts with comparing the mean of your sample data ($\bar{x}$) against a broader context, often called the 'population of means'. This comparison is crucial because, according to the Central Limit Theorem, the distribution of the sample mean approaches a normal (Gaussian) distribution as the sample size increases. This principle holds true even for smaller sample sizes through the use of a t-test, highlighting the versatility of hypothesis testing across various types of data, from rainfall to test scores, and beyond.

Let's walk through a real-life scenario to illustrate the process:

Imagine your friend Jake claims, "The average rainfall here is 3.4396 cm per day in spring." To challenge this claim with a hypothesis test, you would:

1. Collect daily rainfall samples during spring at the specified location.
2. Calculate the mean of these samples.
3. Start with the assumption that Jake's claim is correct (Null Hypothesis H0: $\mu$ = 3.4396 cm, Alternative Hypothesis H1: $\mu$ ≠ 3.4396 cm), where "≠" signifies your goal is to check if Jake's claim could be incorrect. If you aimed to prove the rainfall is more, H1 would be $\mu$ > 3.4396 cm.
4. Choose a significance level (alpha), such as 0.1, 0.05, or 0.01, setting the standard for how strong your evidence must be to convincingly dispute Jake's claim.
5. Calculate the test statistic (either z-score or t-score), which depends on your sample size and whether you know the population standard deviation. As a general guideline, use a z-score for sample sizes over 29 or when the population standard deviation is known. Otherwise, use a t-score.
6. Find the p-value associated with your test statistic. This step involves comparing your test statistic to a critical value (or values, in the case of a two-tailed test), which can either be a t-score or z-score.
7. Compare the p-value with your chosen alpha level. If the p-value is less than alpha, Jake's claim is statistically disputed (you reject the null hypothesis). If the p-value is greater, you haven't disproven Jake's claim, but this doesn't confirm he's right; it simply means you haven't found enough evidence to reject the null hypothesis.

This process, grounded in statistical theory, empowers you to make informed decisions and draw conclusions from data, making hypothesis testing a cornerstone of data analysis across disciplines.

## Demystifying Confidence Intervals

Confidence intervals offer an alternative yet insightful perspective to hypothesis testing. While hypothesis testing leads to conclusions like "I have rejected the hypothesis that the average wind speed is 20 km/hr with a significance level of alpha = 0.05," confidence intervals frame this information differently. For instance, a confidence interval might declare, "I am 95% confident that the true average wind speed is between 15 and 19 km/hr." Noticeably, if the hypothesized value of 20 km/hr does not fall within this interval, it implies that, under a hypothesis test with alpha = 0.05, we would reject the null hypothesis that the mean wind speed is 20 km/hr.

So, why do we use both hypothesis testing and confidence intervals? The reasons are varied, but a key distinction lies in their application: hypothesis testing is primarily about validating or refuting specific claims. In contrast, confidence intervals provide a range within which we believe the true parameter value lies, offering a measure of uncertainty without necessarily making a direct claim about its accuracy.

Confidence intervals serve a critical role in statistical analysis by not only supporting or challenging a hypothesis indirectly but also by giving a quantitative estimate of the precision of an estimate and the degree of uncertainty associated with it. This dual functionality enriches our understanding and interpretation of data, allowing for a more nuanced view of the statistical landscape.

## Python Guide for Handling Data and Statistical Analysis

### Reading Data from Text Files

#### Note: We have conducted all the required input/output operations for this homework in problem2.py and problem3.py. The additional information provided below is for your reference, should you wish to explore further.

In the realm of data formats, you'll encounter various types, including .csv, .json, and .txt files. This assignment focuses on text (.txt) files, offering a straightforward method to access and manipulate data. Here’s a step-by-step approach to work with text files in Python:

1. **Opening a Text File**: To begin, open your text file in read mode. This step is crucial for accessing the data within.

```python
myFile = open('sample.txt')
```

2. **Reading Data**: Next, read the contents of the file. Using `.readlines()` will store each line of the file as a separate string element in a list, making it easier to process the data line by line.

```python
data = myFile.readlines()
```

3. **Closing the File**: Always remember to close the file once you've finished reading its contents. This practice is essential for resource management and to avoid potential errors in file handling.

```python
myFile.close()
```

After loading the data, you'll have a list where each element corresponds to a line from the file, stored as a string.

4. **Converting Strings to Numbers**: Since the data is in string format, you'll likely need to convert these strings into a numeric format (like floats) for analysis. This conversion can be efficiently performed using a list comprehension:

```python
data = [float(x) for x in data]
```

This step transforms your data into a more useful form for numerical operations.

### Utilizing `numpy` for Statistical Calculations

While manual calculations of statistical measures are possible, leveraging Python's `numpy` library can significantly simplify these tasks:

- **Importing `numpy`**: Begin by importing the library to access its comprehensive suite of functions.

```python
import numpy as np
```

- **Calculating the Mean**: The mean, or average, gives you a central value of your data. `numpy` provides a straightforward method to calculate this:

```python
avg = np.mean(data)
```

- **Determining the Standard Deviation**: The standard deviation measures the amount of variation or dispersion in your data set. `numpy` also simplifies this calculation, but you need to be mindful of the differential degrees of freedom (`ddof` parameter):

```python
sd = np.std(data, ddof=x)
```

Here, `x` should be set based on your data context:
- Use `ddof=1` for sample data, which adjusts the calculation to use `N-1` (`N` = No. of samples) as the denominator, providing an unbiased estimate of the population standard deviation.
- Use `ddof=0` for population data, where `N` is the appropriate denominator since you're considering the entire data set.


### Utilizing Standard Normal and Student's t Distributions in Python

In your homework, two pivotal distributions will be frequently encountered: the `standard normal (z)` and the `Student's t` distributions. These distributions are instrumental in conducting hypothesis testing and calculating confidence intervals.

#### Working with the Standard Normal Distribution

1. **Importing the Distribution**: To use the standard normal distribution, first import it from `scipy.stats`:

```python
from scipy.stats import norm
```

2. **Calculating Probabilities**: To determine the probability that a random variable under the standard normal distribution falls below a specific value `z_c`, use the cumulative distribution function (cdf):

```python
p = norm.cdf(z_c)
```

3. **Finding Specific Points**: Conversely, if you need to find the value `z_c` below which a certain proportion `p` of observations fall (the inverse of the cdf), you can use:

```python
z_c = norm.ppf(p)
```

These functions allow you to navigate the standard normal distribution, enabling you to calculate probabilities and critical values essential for statistical analysis.

#### Leveraging the Student's t Distribution

1. **Importing the Distribution**: Similar to the normal distribution, start by importing the Student's t distribution from `scipy.stats`:

```python
from scipy.stats import t
```

2. **Calculating Probabilities with Degrees of Freedom**: To find the probability of a value lying below a certain point `t_c` in the t distribution, the degrees of freedom (`df`) play a crucial role:

```python
p = t.cdf(t_c, df)
```

The `df` parameter is vital as it adjusts the distribution based on the sample size, taking into account the estimation of the mean.

3. **Inverse Calculations for Critical Values**: To identify the critical value `t_c` for a given probability `p`, similar to the norm, you use the inverse cdf, ensuring to include the degrees of freedom:

```python
t_c = t.ppf(p, df)
```

The degrees of freedom for the t distribution typically are `N-1`, acknowledging that one degree of freedom is used for estimating the sample mean. This adjustment is crucial for accurately representing the distribution when the population standard deviation is unknown, making the Student's t distribution a reliable tool for smaller sample sizes.


<!-- ## Very Important Notes for Your Analysis: -->
### Understanding the Data Sources

You will engage with data from three distinct sampled groups, each offering a unique perspective on vehicle ages:

1. **City Vehicle Age Survey**: Comprehensive data were gathered on the average age of vehicles across different counties in different states. This dataset is stored in `city_vehicle_survey.txt`, with each line representing the average age of vehicles for a specific county. This foundational dataset offers a broad overview of vehicle ages in different states.

2. **Focused Sampling on Emission Control Program Impact**: To gain deeper insights, additional targeted sampling was conducted. Two further datasets were compiled to explore the effect of emission control programs in different states:
   - The average vehicle ages with active emission control programs are documented in `vehicle_data_1.txt`.
   - The average vehicle ages without such programs are recorded in `vehicle_data_2.txt`.

### Your Analytical Objectives

Your task is to apply hypothesis testing techniques to address key questions such as:

   - "Is the average age of vehicles 5 years?"
   - "Is there a discernible difference in the average vehicle ages with emission control programs and those without?"


# Instructions

## Step-by-Step Guide to Complete the Homework

This homework is designed to enhance your understanding and application of statistical analysis techniques through practical experience with real-world data. You'll be working with survey data on vehicle ages to assess the impact of emission control programs on the average age of vehicles in different counties. Below is a structured approach to guide you through the project.

### 0) Repository Setup

First, ensure your repository is correctly set up with the following files, crucial for Problem 2:

1. `city_vehicle_survey.txt`: Contains data on the average age of vehicles across various counties.
2. `vehicle_data_1.txt`: Similar data for counties with active emission control programs.
3. `vehicle_data_2.txt`: Data for counties without emission control programs.
4. `Writeup.pdf`: This is the file you'll submit to GradeScope. This document is a PDF with fillable fields, you are supposed to enter your response in this PDF file. 

**Important Note**: When calculating the standard error in both Problem 2 and Problem 3, remember that the true population mean is unknown. Thus, when using `np.std(data, ddof=x)`, set `x` to `1` to account for this uncertainty.

### 1) Problem 1: Sampling Techniques

This problem has no coding component. Refer to the instructions in the Writeup.pdf file for this question; ensure you save the PDF after entering your responses.

### 2) Problem 2: Hypothesis Testing

Your tasks involve answering specific questions in the Writeup.pdf for GradeScope submission and using the provided `problem2.py` file for your code solutions.

This segment focuses on analyzing vehicle age datasets within `city_vehicle_survey.txt`, `vehicle_data_1.txt`, and `vehicle_data_2.txt`:

1. **Formulating Hypotheses**: The transportation department believes that the average age of vehicles is *5 years*. You'll need to challenge this assertion by formulating null and alternative hypotheses and deciding on the appropriate statistical test.

2. **Statistical Analysis**: Execute a statistical test with the `city_vehicle_survey.txt data. Detail the sample size, mean, standard error, test score (z or t), and p-value. Evaluate the significance of your results at 0.05 and 0.10 significance levels and discuss any conclusions you can draw at these confidence levels.

3. **Error and Sample Size Calculation**: Determine the largest standard error that still results in a significant test at a significance level of 0.05, and the corresponding minimum sample size needed for this condition.

4. **Comparative Analysis**: Assess whether there is a difference in the mean vehicle ages between counties with and without emission control programs. Formulate your null and alternative hypotheses accordingly, and select the suitable test type for this analysis.
5. **Executing the Two-sample Test**: Analyze the data from `vehicle_data_1.txt` and `vehicle_data_2.txt`. Report the sample sizes, means, standard errors, z-scores, and p-values. Determine the significance of your findings at 0.05 and 0.10 levels, using a two-sample z-test, and conclude what, if anything, can be inferred about your hypotheses at these confidence intervals.

## 2) Problem 3: Confidence Intervals 

Answer the following questions on the Writeup.pdf - (Reminder! you will upload Writeup.pdf to Gradescope)
Use and submit the given problem3.py file for your code thorugh GitHub as usual.

You are collaborating with a nutritionist to assess the average sodium content (in milligrams) of a new brand of health snack bars. The following dataset represents average sodium of the snacks in different factories:
`[135, 140, 130, 145, 150, 138, 142, 137, 136, 148, 141, 139, 143, 147, 149, 134, 133, 146, 144, 132]`


1. Use the sample to construct a 90% confidence interval for the average sodium of the snacks. Report whether you will use a z-test or t-test and report the sample mean, the standard error, the standard statistic (t or z value), and the interval. (Think, which distribution should you use here if very few datapoints are available?)

2. Repeat Q1 for a 95% confidence interval. What is the standard statistics value (t or z value) and what is the interval? Is your interval wider or narrower compared to using the 90% confidence interval?

3. Repeat Part 2 if you are told that the population standard deviation is `5`. (Think, which distribution should you use here now that you have the true population standard deviation?). Report whether you will use a z-test or t-test and the values for the sample mean, standard error, standard statistic, and confidence interval. Is your interval wider or narrower than the interval computed in Q2?

# Regarding writing the responses
Round values to four decimal places after the decimal point. For values less than $0.01$, express them in scientific notation. For instance:

1. Use $1.0000 \times 10^{-3}$ instead of $0.001$.
2. Use $4.5000 \times 10^{-4}$ instead of $0.00045$.
3. Use $7.2727 \times 10^{-5}$ instead of $0.00007272727$.

# Submitting your code to GitHub and the writeup on Gradescope

Please commit and push the latest version of your code files  as you have done in previous assignments. Please verify your submitted files by looking at GitHub online. 

At https://www.gradescope.com/, after logging in with your purdue email. You should see the assignment named "Homework 5" where you will upload your completed Writeup.pdf (It would be good to confirm you know how to do this before the due date).

And again, Writeup.pdf is a .pdf with fillable fields, make sure to save the .pdf once you have entered your answers!