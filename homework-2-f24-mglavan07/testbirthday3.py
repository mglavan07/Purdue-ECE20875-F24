from homework2 import combine_birthday_data

person_to_day = [("John", 5), ("Jane", 10), ("Mike", 20), ("Lucy", 23), ("Sam", 6)]
person_to_month = [("John", 3), ("Jane", 5), ("Mike", 5), ("Lucy", 3), ("Sam", 10)]
person_to_year = [("John", 1990), ("Jane", 1995), ("Mike", 2000), ("Lucy", 2002), ("Sam", 2023)]

output = combine_birthday_data(person_to_day, person_to_month, person_to_year)
print(output)

#Expected Output: {3: [('John', 5, 1990, 34), ('Lucy', 23, 2002, 22)], 5: [('Jane', 10, 1995, 29), ('Mike', 20, 2000, 24)], 10: ('Sam', 6, 2023, 1)}
