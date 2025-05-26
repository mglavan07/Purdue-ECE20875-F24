import re


def problem1(searchstring):
    """
    Match emails.

    :param searchstring: string
    :return: True or False
    """

    # define a pattern
    p = re.compile(r'^[1-7][0-9]{2}\.[a-zA-Z]{1,10}[0-9]*@(shield.gov|starkindustries.com)$')

    # determine if a match was found
    if p.search(searchstring):

        # if match return "valid"
        return 'valid'
    
    # if no match return "invalid"
    return 'invalid'


def problem2(searchstring):
    """
    Extract author and book.

    :param searchstring: string
    :return: tuple
    """
    # define a pattern
    p = re.compile(r'((\s?[A-Z][a-zA-Z]*\s?){1,2})(wrote)(\sbooks|(\s?[A-Z0-9][A-Z0-9a-z]*\s?){1,3})')

    # determine if a match exists
    if p.search(searchstring):

        # if match start by extracting the text in groups
        author_name = p.search(searchstring).group(1)
        book_name = p.search(searchstring).group(4)

        # remove whitespace from outside 
        author_name = author_name.strip()
        book_name = book_name.strip()

        # return the tuple of information
        return (author_name, book_name)
    
    # if no match return the following tuple
    return ("noauthor", "noname")


def problem3(searchstring):
    """
    Replace Boy/Girl or boy/girl with Man/Woman.

    :param searchstring: string
    :return: string
    """
    # define a pattern 
    p = re.compile(r'(\s?[A-Z][a-z]*\s?)(boy|Boy|girl|Girl)')

    # if a match was found perform an appropriate substitution
    if p.search(searchstring):

        # extract the matched text and determine if Man or Woman
        m = p.search(searchstring).group(2).lower()

        # if boy replace with man
        if m == "boy":
            searchstring = p.sub(r'\1 Man', searchstring)

        # else replace woman
        else:
            searchstring = p.sub(r'\1 Woman', searchstring)

        # remove inside whitespace
        searchstring = searchstring.split(' ')

        # remove whitespace words
        k = 0

        # while loop because the size of the list will shrink
        while k < len(searchstring):

            # remove a '' if it is found
            if searchstring[k] == '':
                removed = searchstring.pop(k)

            # only increment if no pop because the pop will progress the pointer
            else:
                k += 1

        # concactenate the string
        searchstring = ' '.join(searchstring)

        # return the updated string
        return searchstring.strip()
    
    # if no match return the following string
    return "nomatch"


if __name__ == '__main__':

    print("\nProblem 1:")
    testcase11 = '123.iamironman@starkindustries.com'
    print("Student answer: ",problem1(testcase11),"\tAnswer correct?", problem1(testcase11) == 'valid')

    testcase12 = '250.Srogers1776@starkindustries.com'
    print("Student answer: ",problem1(testcase12),"\tAnswer correct?", problem1(testcase12) == 'valid')

    testcase13 = '100.nickfury@shield.gov'
    print("Student answer: ",problem1(testcase13),"\tAnswer correct?", problem1(testcase13) == 'valid')

    testcase14 = '144.venom@starkindustries.comasdf'
    print("Student answer: ",problem1(testcase14),"\tAnswer correct?", problem1(testcase14) == 'invalid')

    testcase15 = '942.hyperion@starkindustries.com'
    print("Student answer: ",problem1(testcase15),"\tAnswer correct?", problem1(testcase15) == 'invalid')

    testcase16 = '567.greengoblin@shield.gov'
    print("Student answer: ",problem1(testcase16),"\tAnswer correct?", problem1(testcase16) == 'invalid')

    testcase17 = '324drdoom324@starkindustries.com'
    print("Student answer: ",problem1(testcase17),"\tAnswer correct?", problem1(testcase17) == 'invalid')

    testcase18 = '765.Hosborn*876@shield.gov'
    print("Student answer: ",problem1(testcase18),"\tAnswer correct?", problem1(testcase18) == 'invalid')

    testcase19 = '234.vulture@shield.com'
    print("Student answer: ",problem1(testcase19),"\tAnswer correct?", problem1(testcase19) == 'invalid')


    print("\nProblem 2:")
    testcase21 = "George Orwell wrote 1984"
    print("Student answer: ",problem2(testcase21),"\tAnswer correct?", problem2(testcase21) == ("George Orwell","1984"))

    testcase22 = "In the 1930s, a Mystery writer wrote Mary Westmacotts. Later it was found that Agatha Christie wrote The Westmacott Novels"
    print("Student answer: ",problem2(testcase22),"\tAnswer correct?", problem2(testcase22) == ("Agatha Christie", "The Westmacott Novels"))

    testcase23 = "Roxette wrote books"
    print("Student answer: ", problem2(testcase23), "\tAnswer correct?", problem2(testcase23) == ("Roxette", "books"))

    testcase24 = "Erin Morgenstern wrote The Starless Sea Book and The Night Circus"
    print("Student answer: ",problem2(testcase24),"\tAnswer correct?", problem2(testcase24) == ("Erin Morgenstern", "The Starless Sea"))

    testcase25 = "Haruki Murakami wrote 1Q84"
    print("Student answer: ",problem2(testcase25),"\tAnswer correct?", problem2(testcase25) == ("Haruki Murakami", "1Q84"))

    testcase26 = "Khaled Hosseini wrote sad books"
    print("Student answer: ",problem2(testcase26),"\tAnswer correct?", problem2(testcase26) == ("noauthor", "noname"))

    testcase27 = "Haruki Murakami wrote Norwegian Wood"
    print("Student answer: ",problem2(testcase27),"\tAnswer correct?", problem2(testcase27) == ("Haruki Murakami", "Norwegian Wood"))


    print("\nProblem 3:")
    testcase31 = 'Spider Boy, I need help!'
    print("Student answer: ",problem3(testcase31),"\tAnswer correct?", problem3(testcase31) == "Spider Man, I need help!")

    testcase32 = 'There is a boy trapped in a burning building Iron Boy'
    print("Student answer: ",problem3(testcase32),"\tAnswer correct?", problem3(testcase32) == "There is a boy trapped in a burning building Iron Man")

    testcase33 = 'Spider Girl, I need help!'
    print("Student answer: ",problem3(testcase33),"\tAnswer correct?", problem3(testcase33) == "Spider Woman, I need help!")

    testcase34 = 'The Invisible girl is a member of the Fantastic Four'
    print("Student answer: ",problem3(testcase34),"\tAnswer correct?", problem3(testcase34) == "The Invisible Woman is a member of the Fantastic Four")

    testcase35 = 'There is a boy that needs to be saved from the alien!'
    print("Student answer: ",problem3(testcase35),"\tAnswer correct?", problem3(testcase35) == "nomatch")
