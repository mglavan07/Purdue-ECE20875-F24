from homework2 import combine_birthday_data

person_to_day = [("Alice", 2), ("Bob", 3), ("Charlie", 1)]
person_to_month = [("Alice", 3), ("Bob", 4), ("Charlie", 3)]
person_to_year = [("Alice", 1980), ("Bob", 1970), ("Charlie", 1990)]

output = combine_birthday_data(person_to_day, person_to_month, person_to_year)
print(output)

#Expected Output: {3: [('Alice', 2, 1980, 44), ('Charlie', 1, 1990, 34)], 4: ('Bob', 3, 1970, 54)}
