import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from ReadData import *
from SemiAERating_1 import SemiAERating_1
from AERating_2 import AERating_2
import tensorflow as tf

'''1 Semi-AutoEncoder'''
def RatingPrediction_1(dataset, train_ratio,hidden_neuron,train_epoch):

    random_seed = 1
    batch_size = 500  # 1000 #256 #1024
    lr = 1e-3  # learning rate
    optimizer_method = 'Adam'
    display_step = 1
    decay_epoch_step = 0
    lambda_value = 1
    f_act = tf.identity
    g_act = tf.nn.sigmoid

    if dataset == "MovieTweetings-10k":
        num_users = 123
        num_items = 3096
        num_total_ratings = 2223
        num_i_features = 45  # unprocess
        # desity = 0.59%
        # We only retain users who have rated at least 10 items
    elif dataset == "MovieLens-100k":
        num_users = 943
        num_items = 1682
        num_total_ratings = 100000
        num_i_features = 35
        # desity = 6.30%
    elif dataset == "MovieLens-1m":
        num_users = 6040
        num_items = 3883  # 3952
        num_total_ratings = 1000209
        num_i_features = 35
        # desity = 4.26%

    R, mask_R, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
    user_train_set,item_train_set,user_test_set,item_test_set, train_R_4_test, train_mask_R_4_test \
        = read_rating_1(num_users, num_items, num_i_features, num_total_ratings, train_ratio,dataset)

    with tf.Session() as sess:
        SARating = SemiAERating_1(sess,num_users,num_items, num_i_features, hidden_neuron,f_act,g_act,
                             train_R, train_mask_R, test_R, test_mask_R, train_R_4_test, train_mask_R_4_test,num_train_ratings,num_test_ratings,
                             train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                             decay_epoch_step,lambda_value, user_train_set, item_train_set, user_test_set, item_test_set,R,mask_R)

        Semi_Rating, Rating_R, Rating_mask_R, Rating_test_R, Rating_test_mask_R, num_test_ratings, RMSE_1, MAE_1  = SARating.execute()
    return Semi_Rating, Rating_R, Rating_mask_R, Rating_test_R, Rating_test_mask_R, num_test_ratings, RMSE_1, MAE_1

'''2 AutoEncoder'''
def RatingPrediction_2(index,hidden_neuron,train_epoch,dataset,train_ratio_1,hidden_neuron_1,train_epoch_1):

    train_ratio_2 = 1

    random_seed = 1
    batch_size = 500  # 1000 #256 #1024
    lr = 1e-3  # learning rate
    optimizer_method = 'Adam'
    display_step = 1
    decay_epoch_step = 0
    lambda_value = 1
    f_act = tf.identity
    g_act = tf.nn.sigmoid

    if dataset == "MovieTweetings-10k":
        num_users = 123
        num_items = 3096
        num_total_ratings = 2223
        num_i_features = 45  # unprocess
        # desity = 0.59%
        # We only retain users who have rated at least 10 items
    elif dataset == "MovieLens-100k":
        num_users = 943
        num_items = 1682
        num_total_ratings = 100000
        num_i_features = 35
        # desity = 6.30%
    elif dataset == "MovieLens-1m":
        num_users = 6040
        num_items = 3883  # 3952
        num_total_ratings = 1000209
        num_i_features = 35
        # desity = 4.26%

    Semi_Rating ,Rating_R, Rating_mask_R, Rating_test_R, Rating_test_mask_R , num_test_ratings, RMSE_1, MAE_1 =RatingPrediction_1(dataset, train_ratio_1,hidden_neuron_1,train_epoch_1)

    R, mask_R, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings,num_test_ratings,\
    user_train_set, item_train_set, user_test_set, item_test_set, train_R_4_test, train_mask_R_4_test \
        = read_rating_2( num_users, num_items, Semi_Rating, Rating_mask_R, Rating_R, num_total_ratings, train_ratio_2, index, hidden_neuron,train_epoch, dataset, Rating_test_R, Rating_test_mask_R,num_test_ratings, train_ratio_1, hidden_neuron_1, train_epoch_1)

    with tf.Session() as sess:
        ARating = AERating_2(sess,num_users,num_items, hidden_neuron,f_act,g_act,
                             train_R, train_mask_R,  test_R, test_mask_R, train_R_4_test, train_mask_R_4_test,num_train_ratings,num_test_ratings,
                             train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                             decay_epoch_step,lambda_value, user_train_set, item_train_set, user_test_set, item_test_set)
        res_RMSE, res_MAE = ARating.execute()
        return res_RMSE, res_MAE, RMSE_1, MAE_1



if __name__ == '__main__':
    DataSets_List = ["MovieTweetings-10k", "MovieLens-100k", "MovieLens-1m"]
    dataset = DataSets_List[0]

    # hidden_neuron_1 = 1  # 1000
    # train_epoch_1 = 1  #

    # run_times = 5
    # train_ratio_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    # hidden_neuron_list = [100, 125, 150, 175, 200,225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800]
    # train_epoch_2_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    run_times = 2
    train_ratio_list = [0.5]
    hidden_neuron_list = [100, 125]
    train_epoch_2_list = [5, 6, 7]
    for tempk in range(len(train_ratio_list)):
        for tempi in range(len(hidden_neuron_list)):
            for tempj in range(len(train_epoch_2_list)):
                train_ratio = train_ratio_list[tempk]

                if train_ratio == 0.5:
                    hidden_neuron_1 = 175  # 1000
                    train_epoch_1 = 50  #
                elif train_ratio == 0.6:
                    hidden_neuron_1 = 175  # 1000
                    train_epoch_1 = 6  #
                elif train_ratio == 0.7:
                    hidden_neuron_1 = 275  # 1000
                    train_epoch_1 = 4  #
                elif train_ratio == 0.8:
                    hidden_neuron_1 = 350  # 1000
                    train_epoch_1 = 3  #
                elif train_ratio == 0.9:
                    hidden_neuron_1 = 200  # 1000
                    train_epoch_1 = 5  #

                hidden_neuron_2 = hidden_neuron_list[tempi]
                train_epoch_2 = train_epoch_2_list[tempj]

                result_RMSE = [0] * run_times
                RMSE_ave = []
                RMSE_source = [0] * run_times
                RMSE_sum = 0.0
                result_MAE = [0] * run_times
                MAE_ave = []
                MAE_source =[0] * run_times
                MAE_sum = 0.0

                for i in range(run_times):
                    print("test: ", i + 1)
                    tf.reset_default_graph()
                    result_RMSE[i], result_MAE[i], RMSE_source[i], MAE_source[i]  = RatingPrediction_2(i + 1, hidden_neuron_2, train_epoch_2, dataset, train_ratio,hidden_neuron_1,train_epoch_1)
                    print("test:" + str(i + 1) + " success!")
                    print("==" * 30)

                RMSE_1 = sum(RMSE_source)/len(RMSE_source)
                MAE_1 = sum(MAE_source)/len(MAE_source)

                for j in range(train_epoch_2):
                    for i in range(run_times):
                        RMSE_sum += result_RMSE[i][j]
                        MAE_sum += result_MAE[i][j]
                    RMSE_ave.append(RMSE_sum / run_times)
                    MAE_ave.append(MAE_sum / run_times)
                    RMSE_sum = 0.0
                    MAE_sum = 0.0

                save_path = "./DataSets/"+dataset+"/result/ratio" + str(train_ratio) + "_hidden" + str(hidden_neuron_2) + "_epoch" + str(train_epoch_2) + ".csv"
                save_result_path = "./DataSets/"+dataset+"/result.csv"

                with open(save_path, 'a+', encoding='utf-8') as fb:
                    fb.write("index" + "," + "train_ratio_1" + "," + "hidden_neuron_2" + "," + "train_epoch_2" + "," + "run_times" + "," + "RMSE_1" + "," + "RMSE_ave"+ "," + "MAE_1" + "," + "MAE_ave" + "\n")
                for i in range(len(RMSE_ave)):
                    with open(save_path, 'a+', encoding='utf-8') as fb:
                        fb.write(str(i + 1) + "," + str(train_ratio) + "," + str(hidden_neuron_2) + "," + str(train_epoch_2) + "," + str(run_times) + "," + str(RMSE_1) + "," + str(RMSE_ave[i]) + "," + str(MAE_1) + "," + str(MAE_ave[i]) + "\n")
                with open(save_path, 'a+', encoding='utf-8') as fb:
                    fb.write("**********************************************" + "\n")

                #train_ratio_1, hidden_neuron_2, train_epoch_2, run_times, RMSE_1, RMSE_ave, MAE_1, MAE_ave
                with open(save_result_path, 'a+', encoding='utf-8') as fb:
                    fb.write(str(train_ratio) + "," +str(hidden_neuron_2) +  str(train_epoch_2) + str(run_times) + ",{:.5f}".format(RMSE_1) + ",{:.5f}".format(min(RMSE_ave)) + ",{:.5f}".format(MAE_1) + ",{:.5f}".format(min(MAE_ave)) + "\n")

                print("RMSE_1 = {:.5f}".format(RMSE_1) + "," +"RMSE_average = {:.5f}".format(min(RMSE_ave)) + ","+ "MAE_1 = {:.5f}".format(MAE_1)+ "," + "MAE_average = {:.5f}".format(min(MAE_ave)))
