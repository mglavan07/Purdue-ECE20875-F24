from typing import List, Tuple

def histogram(input_dictionary: dict) -> list:
    # data is a dictionary that contains the following keys: 'data', 'n', 'min_val', 'max_val'
    # n is an integer
    # min_val and max_val are floats
    # data is a list

    # Check for nonzero histogram width
    if input_dictionary["min_val"] == input_dictionary["max_val"]:
        print('Error: min_val and max_val are the same value')
        return []
    
    # Check for empty list
    elif input_dictionary["n"] <= 0:
        return []
    
    # Check for min > max
    elif input_dictionary["min_val"] > input_dictionary["max_val"]:
        c = input_dictionary["min_val"]
        input_dictionary["min_val"] = input_dictionary["max_val"]
        input_dictionary["max_val"] = c

    # If we did not return above, we can make the histogram
    hist = [0] * input_dictionary["n"]
    w = (input_dictionary["max_val"] - input_dictionary["min_val"]) / input_dictionary["n"]

    # iterate through data and increment columns sizes using integer division
    for val in input_dictionary["data"]:

        # ensure the value is between min and max
        if not(val >= input_dictionary["max_val"] or val <= input_dictionary["min_val"]):

            # calculate the bin it belongs to 
            bin = int(val // w)

            # adjust the bin to have the lowest bin be bin = 0 (fixes negative bins)
            adj = abs(int(input_dictionary["min_val"] // w))
            bin += adj

            # increment the bin that was calculated
            hist[bin] += 1
    
    # return the histogram
    return hist 

# Here, the function first checks if the lower and upper bounds are the same, 
# if they are it prints an error message and returns an empty list. 
# If lower bound is greater than upper bound, it swaps their values. 
# If number of bins is less than or equal to 0, it returns an empty list. 
# Then it initializes an empty list hist of length n and calculates the width of each bin. 
# Then it iterates through the data, 
# and for each value checks if it is within the range of the histogram and if it is, 
# it increments the bin it belongs to. Finally, it returns the histogram.

def combine_birthday_data(person_to_day: List[Tuple[str, int]], 
                          person_to_month: List[Tuple[str, int]], 
                          person_to_year: List[Tuple[str, int]]) -> dict:
    # person_to_day, person_to_month, person_to_year are list of tuples

    '''
    person_to_day = [("John", 5), ("Jane", 10), ("Mike", 20), ("Lucy", 23), ("Sam", 6)]
    person_to_month = [("John", 3), ("Jane", 5), ("Mike", 5), ("Lucy", 3), ("Sam", 10)]
    person_to_year = [("John", 1990), ("Jane", 1995), ("Mike", 2000), ("Lucy", 2002), ("Sam", 2023)]
    '''

    # initialize the dictionary
    month_to_people_data = {}

    # iterate through person_to_month to create the dictionary
    for idx, entry in enumerate(person_to_month):

        # assign a name and month
        name, month = entry

        # find remaining age specs
        __, day = person_to_day[idx]
        __, year = person_to_year[idx]
        age = 2024 - year

        # check to see if the month already exists
        found = False
        for key in month_to_people_data:

            # if the key already exists, make a new list of all associated values
            if key == month:
                new = []
                old = [month_to_people_data[key]]

                # add previous names back before adding in the new name 
                for pair in old:
                    new.append(pair)
                new.append((name, day, year, age))

                # overwrite the key pair
                month_to_people_data[key] = new

                # indicate that the month was found so it does not replicated
                found = True

        # if the key was not found, make a new key
        if not found:
            month_to_people_data[month] = (name, day, year, age)
            
    # return the dictionary
    return month_to_people_data

# We first define the current year as 2024, which will be used to calculate the age of each person later on.
# We create an empty dictionary month_to_people_data that will store the final data in the format specified in the problem statement.
# We iterate over the both values in the tuple of the person_to_month list (note that person_to_month is a list of tuples, which means each item in the list is a tuple) 
# using a for loop. For each iteration, we extract the corresponding day, year and age values from the person_to_day and person_to_year lists using the current name as the "key".
# We then use the current month as the key and a tuple of (name, day, year, age) as the value to update the month_to_people_data dictionary.
# Only change the value to a list data type, when there are multiple entries with the same key. This will help append for other tuples to the same month.
# Finally, we return the month_to_people_data dictionary as the output of the function.
