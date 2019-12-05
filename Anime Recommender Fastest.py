import csv
import numpy as np
import itertools
from collections import Counter
import os


def generate_related_anime_recommendations(name, quantity):
    generate_count = 0
    idx = -1
    recommendations = []
    
    try:
        temp_anime_id = anime_data_np[anime_names.index(name), 0]
    except ValueError:
        return recommendations
    
    while generate_count != quantity:
        try:
            idx = anime_1.index(temp_anime_id, idx + 1)
        except ValueError:
            break

        recommendations.append(lift_np[idx, 1])
        generate_count += 1

    return recommendations


def generate_user_based_anime_recommendations(u_id, quantity):
    recommendations = []
    user_index = formatted_users_id_to_index[u_id]

    for anime in formatted_users_data[user_index]:
        if formatted_users_anime_ratings[u_id][anime] == 10:
        
            generate_count = 0
            idx = -1

            while generate_count != quantity:
                try:
                    idx = anime_1.index(anime, idx + 1)
                except ValueError:
                    break

                recommended_anime = lift_np[idx, 1]

                if not recommended_anime in formatted_users_data[user_index]:
                    recommendations.append(recommended_anime)
                    generate_count += 1

    return recommendations


if __name__ == "__main__":

    if os.path.exists("Users_Data.npy"):
        users_data_np = np.load('Users_Data.npy')

    else:
        with open('rating.csv', encoding = "utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            start_reading = False

            users_data = []

            for row in csv_reader:
                if start_reading:
                    if int(row[2]) > 4:
                        users_data.append(row)
                else:
                    start_reading = True

            users_data_np = np.array(users_data)
            np.save("Users_Data", users_data_np)

    users_id = list(users_data_np[:, 0])
    users_anime_id = list(users_data_np[:, 1])

    if os.path.exists("Anime_Data.npy"):
        anime_data_np = np.load('Anime_Data.npy')

    else:
        with open('anime.csv', encoding = "utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            start_reading = False

            anime_data = []

            for row in csv_reader:
                if start_reading:
                    if row[0] in users_anime_id:
                        anime_data.append([row[0], row[1]])
                else:
                    start_reading = True

            anime_data_np = np.array(anime_data)
            np.save("Anime_Data", anime_data_np)

    print("Dataset Loaded")

    anime_names = anime_data_np[:, 1].tolist()
    anime_ids = anime_data_np[:, 0].tolist()
    
    formatted_users_data = []
    formatted_users_anime_ratings = {}
    formatted_users_id_to_index = {}

    temp_user = "-1"
    temp_anime = []
    temp_user_anime_ratings = {}

    for i in range(users_data_np.shape[0]):
        if temp_user != users_id[i]:
            if temp_user != "-1":
                formatted_users_data.append(temp_anime)
                formatted_users_anime_ratings[temp_user] = temp_user_anime_ratings
                formatted_users_id_to_index[temp_user] = len(formatted_users_data) - 1

            temp_user = users_id[i]
            temp_anime = []
            temp_user_anime_ratings = {}

        temp_anime.append(users_anime_id[i])
        temp_user_anime_ratings[users_anime_id[i]] = int(users_data_np[i, 2])

        if i == users_data_np.shape[0] - 1:
            formatted_users_data.append(temp_anime)
            formatted_users_anime_ratings[temp_user] = temp_user_anime_ratings
            formatted_users_id_to_index[temp_user] = len(formatted_users_data) - 1

    print("Dataset Formatted")

    if os.path.exists("Lift.npy"):
        lift_np = np.load('Lift.npy')
        print("Lift Loaded")

    elif os.path.exists("Lift_Text.txt"):
        lift_np = np.loadtxt('Lift_Text.txt', dtype = 'U25')
        print("Lift Loaded")

    else:
        total_users = len(formatted_users_data)

        support_dict = {}

        occurances_dict = Counter(users_anime_id)

        for id_anime in anime_ids:
            support_dict[id_anime] = occurances_dict[id_anime] / total_users

        print("Support Calculated")

        if os.path.exists("Confidence.npy"):
            confidence = np.load('Confidence.npy')
            print("Confidences Loaded")

        elif os.path.exists("Confidence_Text.txt"):
            confidence = np.loadtxt('Confidence_Text.txt', dtype = 'U25')
            print("Confidence Loaded")

        else:
            anime_combinations = list(itertools.combinations(anime_ids, 2))

            print("Anime Combinations Generated")

            confidence_counter = Counter()

            for counter, data in enumerate(formatted_users_data):
                temp = itertools.combinations(data, 2)
                temp2 = []

                for A1, A2 in temp:
                    temp2.append(A1 + " " + A2)

                confidence_counter.update(temp2)

                print("Total Users:", total_users, " Current User:", counter + 1)

            print("Combinations For Confidences Generated And Counted")
            
            min_confidence_threshold = 0.2
            
            total_combinations = len(anime_combinations)

            confidence = np.zeros(shape = (total_combinations, 3), dtype = "U25")

            index = 0

            for counter, combo in enumerate(anime_combinations):
            
                occurances = confidence_counter[combo[0] + " " + combo[1]] + confidence_counter[combo[1] + " " + combo[0]]
              
                if occurances != 0:
                    
                    temp_confidence_A0_A1 = occurances / occurances_dict[combo[0]]
                    temp_confidence_A1_A0 = occurances / occurances_dict[combo[1]]
                    
                    if temp_confidence_A0_A1 >= min_confidence_threshold:
                        confidence[index][0] = combo[0]
                        confidence[index][1] = combo[1]
                        confidence[index][2] = str(temp_confidence_A0_A1)

                        index += 1
                    
                    if temp_confidence_A1_A0 >= min_confidence_threshold:
                        confidence[index][0] = combo[1]
                        confidence[index][1] = combo[0]
                        confidence[index][2] = str(temp_confidence_A1_A0)

                        index += 1

                print("Total Combinations:", total_combinations, " Current Combination:", counter + 1)

            confidence = confidence[:index]
            
            print("Confidences Calculated")

            np.save('Confidence', confidence)
            np.savetxt('Confidence_Text.txt', confidence, fmt = "%s")

            print("Confidences Saved")

        lift_np = np.zeros_like(confidence, dtype = "U25")
        
        min_lift_threshold = 1
        
        index = 0
        
        for counter, conf in enumerate(confidence):
            temp_lift_value = float(conf[2]) / support_dict[conf[1]]
            
            if temp_lift_value > min_lift_threshold:
                lift_np[index][0] = conf[0]
                lift_np[index][1] = conf[1]
                lift_np[index][2] = str(temp_lift_value)
                index += 1

            print("Total Values:", lift_np.shape[0], " Current Value:", counter + 1)
        
        lift_np = lift_np[:index]

        lift_np = lift_np[np.argsort(lift_np[:, 2].astype(np.float64))[::-1]]
        
        print("Lift Calculated And Sorted")

        np.save('Lift', lift_np)
        np.savetxt('Lift_Text.txt', lift_np, fmt = "%s")

        print("Lift Saved")

    anime_1 = lift_np[:, 0].tolist()

    anime_list_for_recommendations = ["Naruto", "Gintama°", "Bakuten Shoot Beyblade G Revolution", "Dragon Ball Z"]

    for anime in anime_list_for_recommendations:
        print()
        print("Generating Related Anime Based Recommendations For", anime)
        print()

        generated_recommendations = generate_related_anime_recommendations(anime, 10)

        for counter, recommend in enumerate(generated_recommendations):
            print("Recommendation No.", counter + 1, " ", anime_data_np[anime_ids.index(recommend), 1])

            
    user_list_for_recommendations = ["112"]

    for user in user_list_for_recommendations:
        print()
        print("Generating User Based Recommendations For", user)
        print()

        generated_recommendations = generate_user_based_anime_recommendations(user, 5)

        for counter, recommend in enumerate(generated_recommendations):
            print("Recommendation No.", counter + 1, " ", anime_data_np [anime_ids.index(recommend), 1])