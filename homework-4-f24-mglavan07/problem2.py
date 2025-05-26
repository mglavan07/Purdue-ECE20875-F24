def stencil(data, f, width):
    """
    1) perform a stencil using the filter function f with 'width', on list data.
    2) return the resulting list output.
    3) note that if len(data) is k, len(output) would be k - width + 1.
    4) f will accept input a list of size 'width' and return a single number.

    :param data: list
    :param f: function
    :param width: int
    :return output: list
    """
    # initialize the stencil data set
    sten = []

    # iterate from 0 to the length of the stencil
    for i in range(len(data) - width + 1):

        # slice the window
        win = data[i : i + width]

        # use the filter function
        filt = f(win)

        # append the filtered window to stencil
        sten.append(filt)

    # return the stenciled list
    return sten

def create_box(box):
    """
    1) This function takes in a list, box.
    The box_filter function defined below accepts a list L of length len(box) and returns a simple
    convolution of it with the list, box.

    2) The meaning of this box filter is as follows:
    for each element of input list L, multiply L[i] by box[len(box) - 1  - i],
    sum the results of all of these multiplications and return the sum.

    3) For a box of length 3, box_filter(L) should return:
      (box[2] * L[0] + box[1] * L[1] + box[0] * L[2]),
      similarly, for a box of length 4, box_filter should return:
      (box[3] * L[0] + box[2] * L[1] + box[1] * L[2] + box[0] * L[3])

    The function create_box returns the box_filter function, as well as the length
    of the input list box

    :param box: list
    :return box_filter: function, len(box): int
    """

    def box_filter(L):

        # check if L is the same as len(box)
        if len(L) != len(box):
            print(f'Calling box filter with the wrong length list. Expected length of list should be %d.', len(box))
            return 0

        # perform the convolution
        else:

            # intialize an empty variable for the convolution value
            conv = 0
            
            # iterate through L and multiply by the inversely parallel box values
            for i in range(len(box)):

                # one L_i * B_end-i for the convolution
                conv += L[i] * box[len(box) - i - 1]

            # return the calculated value
            return conv

    # return the function and length
    return box_filter, len(box)


if __name__ == '__main__':

    def mov_avg(L):
        return float(sum(L)) / 3

    def sum_sq(L):
        return sum([i**2 for i in L])

    data = [1, 3, 5, 7, -9, -9, 11, 13, 15, 17, 19]

    print(stencil(data, mov_avg, 3))
    print(stencil(data, sum_sq, 5))

    # note that this creates a moving average!
    box_f1, width1 = create_box([1.0 / 3, 1.0 / 3, 1.0 / 3])
    print(stencil(data, box_f1, width1))

    box_f2, width2 = create_box([-0.5, 0, 0, 0.5])
    print(stencil(data, box_f2, width2))
