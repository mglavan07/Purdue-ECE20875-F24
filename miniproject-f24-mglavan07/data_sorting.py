# Functions to be used to structure data for the project
import statistics

def make_video_dictionary(DataFrame):
    '''
    input - pandas DataFrame for the performance data
    
    output - a dictioanry:
      key : pair
      video number : ([student IDs], [feature statistics], ...)
    '''
    # simplify the name of the dataframe
    df = DataFrame

    # empty dictionary
    video_dict = {}

    # get names of colums of interest
    col_names = list(df.columns)
    # print(col_names)
    col_names.remove("VidID")

    # Iterate through the rows of the DataFrame
    for _, row in df.iterrows():
        vid_id = int(row['VidID'])  # Get the VidID (key)

        # If the VidID is not already a key, initialize it with an empty tuple of lists
        if vid_id not in video_dict:
            video_dict[vid_id] = tuple([] for _ in range(len(df.columns) - 1))

        # Append the corresponding values from all other columns to the appropriate lists
        for idx, col_name in enumerate(col_names):
            if col_name != 'VidID':  # Skip the VidID column if seen
                video_dict[vid_id][idx].append(row[col_name])

    # Print the first few items of the dictionary to verify
    # print(video_dict.keys())
    # print(video_dict[50])

    # return the dictionary
    return video_dict

def make_student_dictionary(DataFrame):
    '''
    input - pandas DataFrame for the performance data
    
    output - a dictioanry:
      key : pair
      stident ID : ([watched video ID], [feature statistics], ...)
    '''
    # simplify the name of the dataframe
    df = DataFrame

    # empty dictionary
    student_dict = {}

    # get names of colums of interest
    col_names = list(df.columns)
    col_names.remove("userID")


    # Iterate through the rows of the DataFrame
    for _, row in df.iterrows():
        user_id = str(row['userID'])  # Get the userID (key)

        # If the userID is not already a key, initialize it with an empty tuple of lists
        if user_id not in student_dict:
            student_dict[user_id] = tuple([] for _ in range(len(df.columns) - 1))

        # Append the corresponding values from all other columns to the appropriate lists
        for idx, col_name in enumerate(col_names):
            if col_name != 'userID':  # Skip the userID column if seen
                student_dict[user_id][idx].append(row[col_name])

    # Print the first few items of the dictionary to verify
    # print(video_dict.keys())
    # print(video_dict[50])

    # return the dictionary
    return student_dict

def at_least_n_videos(video_dict, student_dict, n):
    '''
    input - dictionary with keys being the video ID
            dictionary with keys being the student ID
            an integer representing the minimum number of videos watched
    
    output - a dictioanry:
      key : pair
      video ID : ([valid student IDs], [valid feature statistics], ...)
    '''

    # create an empty list for invalid students
    fewer_n = []

    # initialize a new dictionary
    valid_dict = {}

    # identify students under n videos
    for student in list(student_dict.keys()):

        # find how many videos the student watched
        watched = len(student_dict[student][0])

        # if it is less than n, append to the fewer_n
        if watched < n:
            fewer_n.append(student)

    # remove rows with the student ID from video_data
    for video in list(video_dict.keys()):

        # make a new key in the new dictionary
        valid_dict[video] = []

        # find the chunk of data corresponding to the video
        video_data = video_dict[video]

        # make empty sublists in the new dictionary
        valid_dict[video] = [[] for _ in range(len(video_data))]

        # iterate through all the rows in the video_data
        for i, id in enumerate(video_data[0]):

            # add the data from video_data into the new dictionary if the ID is NOT in the fewer_n
            if id not in fewer_n:

                # for each feature
                for j in range(len(valid_dict[video])):

                    # append to the feature, row
                    valid_dict[video][j].append(video_data[j][i])

    # return the dictioanry
    return valid_dict


def normalize_features(data):
    # Initialize an empty list to hold the normalized data
    normalized_data = []

    # Iterate over each row
    for row in data:
        # Calculate the mean and standard deviation for the row
        mean = statistics.mean(row)
        stdev = statistics.stdev(row)
        
        # Normalize the row and add it to the normalized_data list
        normalized_row = [(value - mean) / stdev for value in row]
        normalized_data.append(normalized_row)
    
    return normalized_data

