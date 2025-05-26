from homework2 import combine_birthday_data

person_to_day = [("John", 15), ("Jane", 20), ("Bob", 5)]
person_to_month = [("John", 1), ("Jane", 2), ("Bob", 1)]
person_to_year = [("John", 1985), ("Jane", 1989), ("Bob", 1995)]

output = combine_birthday_data(person_to_day, person_to_month, person_to_year)
print(output)

#Expected Output: {1: [('John', 15, 1985, 39), ('Bob', 5, 1995, 29)], 2: ('Jane', 20, 1989, 35)}
