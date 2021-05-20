import numpy as np
import os
import csv


EXP_NAMES = ["H1_0_F1" ,"H1_0_F2","H1_45_F1","H1_45_F2","H1_90_F1","H1_90_F2" ,"H4_90_F1" ,"H4_90_F2","H4_90_F1_random","H4_90_F2_random" ,"no_strain_F1" ,"no_strain_F2"]



FOLDER_PATHS = {"H1_0_F1" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\M-8-0-H1\\M-8-0-H1 - DIC face 1",
                "H1_0_F2" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\M-8-0-H1\\M-8-0-H1 - DIC face 2",

                "H1_45_F1" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\M-8-45-H1\\M-8-45-H1 - DIC face 1",
                "H1_45_F2" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\M-8-45-H1\\M-8-45-H1 - DIC face 2",

                "H1_90_F1" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\M-8-90-H1\\M-8-90-H1 - DIC face 1",
                "H1_90_F2" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\M-8-90-H1\\M-8-90-H1 - DIC face 2",

                "H4_90_F1" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\M-8-0-H4\\M-8-0-H4\\Data in linear range\\DIC face 1",
                "H4_90_F2" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\M-8-0-H4\\M-8-0-H4\\Data in linear range\\DIC face 2",

                "H4_90_F1_random" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\M-8-0-H4\\M-8-0-H4\\Random error\\DIC face 1",
                "H4_90_F2_random" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\M-8-0-H4\\M-8-0-H4\\Random error\\DIC face 2",

                "no_strain_F1" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\no_strain\DIC face 1",
                "no_strain_F2" : "C:\\Users\\wbald\\Documents\\mech inverse\\fourth_year_project\\Machined coupon data\\no_strain\DIC face 2"}




BLOCK_DIMS = {"H1_0_F1" : np.array([20.01, 120.36, 4.84]),
              "H1_0_F2" : np.array([20.01, 120.36, 4.84]),
                
              "H1_45_F1" : np.array([15.8, 120.52, 4.47]),
              "H1_45_F2" : np.array([15.8, 120.52, 4.47]),
              
              "H1_90_F1" : np.array([20.09, 120.07, 4.18]),
              "H1_90_F2" : np.array([20.09, 120.07, 4.18]),
              
              "H4_90_F1" : np.array([20.09, 117., 4.18]),
              "H4_90_F2" : np.array([21., 115., 4.18]),
              
              "H4_90_F1_random" : np.array([20.09, 117., 4.18]),
              "H4_90_F2_random" : np.array([21., 115., 4.18]),
              
              "no_strain_F1" : np.array([19.5, 128., 5.]),
              "no_strain_F2" : np.array([19.5, 128., 5.])}




BLOCK_AXES_OFFSETS = {"H1_0_F1" : np.array([2.5, -85.]),
                      "H1_0_F2" : np.array([2.5, -85.]),

                      "H1_45_F1" : np.array([-6.5,-87.5]),
                      "H1_45_F2" : np.array([-6.5,-87.5]),

                      "H1_90_F1" : np.array([-7,-95]),
                      "H1_90_F2" : np.array([-7.5, -92]),

                      "H4_90_F1" : np.array([-24, -93]),
                      "H4_90_F2" : np.array([-25, -92]),

                      "H4_90_F1_random" : np.array([-23.75, -94]),
                      "H4_90_F2_random" : np.array([-25, -93]),

                      "no_strain_F1" : np.array([2.5, -88.]),
                      "no_strain_F2" : np.array([2.5, -88.])}



def get_folder_stress(fol_path):
    file_names = os.listdir(fol_path)
    file_name = next((name for name in file_names if name == 'stress.csv'), 'none_found')
    if file_name == 'none_found':
        raise ValueError("didn't find a file stress file in the given folder")
    
    stresses = {}
    with open(fol_path + "\\" + file_name, newline='') as f:
        reader = csv.reader(f)            
        next(reader)
        for row in reader:
            if len(row[0]) == 1:
                stresses['0' + row[0]] = float(row[1])
            else:
                stresses[row[0]] = float(row[1])
            
    return stresses




def unioned_folder_displacement(fol_path):
    
    measurement_coordinates = []
    with open(fol_path + "\\unioned\\" + "coordinates.csv", newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            measurement_coordinates.append(list(float(row[i]) for i in [0, 1]))
    measurement_coordinates = np.asarray(measurement_coordinates)
    
    
    # get a list of file names in the folder with the correct format
    folder_contents = os.listdir(fol_path  + "\\unioned")
    file_names = []
    for file_name in folder_contents:
        if file_name[:11] == "union_data_" and file_name[-4:] == ".csv":
            file_names.append(file_name)            
    print('found ' + str(len(file_names)) + ' data files')


    displacements = {}
    for file_name in file_names:
        with open(fol_path + "\\unioned\\" + file_name, newline='') as f:
            reader = csv.reader(f)
            arr = []
            for row in reader:
                arr.append(list(float(row[i]) for i in [0,1]))
            arr = np.asarray(arr)

        displacements[file_name[-6:-4]] = arr
        

    return measurement_coordinates, displacements





def raw_arrays_from_folder(fol_path, columns):
    """     
        args:
            fol_path - string - absolute path to folder containing the data files
            columns  - list - indices denoting the columns of the spreadsheets to read.
        
        returns:
            data - dictionary - { [file_name] : [array of request columns] , ...  }       
            headers - list - header names corresponding to slected columns
    """
    
    # check columns is sorted to avoid confusion
    if sorted(columns) != columns:
        raise ValueError('input column list shold be sorted')
        
    
    # get a list of file names in the folder with the correct format
    folder_contents = os.listdir(fol_path)
    file_names = []
    for file_name in folder_contents:
        if file_name[:5] == "data_" and file_name[-4:] == ".csv":
            file_names.append(file_name)            
    print('found ' + str(len(file_names)) + ' files')
    
    
    data = {}
    for file_name in file_names:
        with open(fol_path + "\\" + file_name, newline='') as f:
            reader = csv.reader(f)            
            headers = next(reader)
            
            arr = []
            for row in reader:
                arr.append(list(float(row[col]) for col in columns))
            arr = np.asarray(arr)
        
        # remove the .csv in the keys of the dictionary
        data[file_name[:-4]] = arr
      
    headers = [list(headers)[col] for col in columns]

    return data, headers





def print_col_numbers(folder_path):
    file_names = os.listdir(folder_path)

    # look for a file to read the column names from
    # looks for a file with a name of the form 'data_XXXX.csv'
    file_name = next((name for name in file_names if (name[:5] == 'data_' and name[-4:] == '.csv')), 'none_found')
    if file_name == 'none_found':
        raise ValueError("didn't find a file with a name matching the template 'data_XXXX.csv' in the given folder")

    # print [column name] = [index in sheet]
    with open(folder_path + "\\" + file_name, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        c = 0
        for header in headers:
            print(header + " = " + str(c))
            c += 1
            
            
            
            
def union_measurement_points(jagged_input_data):
    """ jagged_input_data is a dictionary of name-array pairs. The first two columns of all arrays should be the coordinates """
    names = [key for key in jagged_input_data.keys()]
    raw_numbers_list = [jagged_input_data[key].copy() for key in names]
    
    counter = 0
    end_early = False
    while counter < raw_numbers_list[0].shape[0]:
        counter_prime = counter
        dset_index = 1
        p0 = raw_numbers_list[0][counter, :2]
        restart_row = False
        while dset_index < len(raw_numbers_list):

            if counter >= len(raw_numbers_list[dset_index]):
                # have reached the end early, delete all remaining rows of dset0 and exit.
                raw_numbers_list[0] = raw_numbers_list[0][:counter]
                end_early = True
                break

            p1 = raw_numbers_list[dset_index][counter, :2]
            if np.allclose(p0, p1, atol=0.001):
                dset_index += 1
            else:
                while counter_prime < raw_numbers_list[dset_index].shape[0]-1:
                    counter_prime += 1

                    # check for early end of list
                    if counter_prime >= len(raw_numbers_list[dset_index]):
                        # exit with no find.
                        counter_prime = raw_numbers_list[dset_index].shape[0]-1
                        break

                    p1 = raw_numbers_list[dset_index][counter_prime, :2]
                    if np.allclose(p0, p1, atol=0.001):
                        raw_numbers_list[dset_index] = np.delete(raw_numbers_list[dset_index], range(counter, counter_prime), axis=0)
                        dset_index += 1
                        break

                if dset_index == len(raw_numbers_list):
                    break

                if counter_prime == raw_numbers_list[dset_index].shape[0]-1:
                    for prev_sets in range(dset_index):
                        raw_numbers_list[prev_sets] = np.delete(raw_numbers_list[prev_sets], counter, axis=0)
                        restart_row = True

            if restart_row:
                break

        if end_early:
            break

        if dset_index == len(raw_numbers_list):
            counter += 1

    # final task is to remove any elements left over past the length of list
    num_matched = raw_numbers_list[0].shape[0]
    raw_numbers_list = [arr[:num_matched] for arr in raw_numbers_list]

    # now strip the coordinates:
    coordinates = raw_numbers_list[0][:, :2]
    datas = [lst[:, 2:] for lst in raw_numbers_list]
    
    # put back into a dictionary...
    data_return = {}
    for index, name in enumerate(names):
        data_return[name] = datas[index]

    return coordinates, data_return


