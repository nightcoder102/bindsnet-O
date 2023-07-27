
from bindsnet.getAccuracy import getAccuracy
from getSTDPParameterFromData import get_STDP_param_from_data
import os
import time
from math import isnan
print('not master')
def create_directory(directory_name='logs'):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        return directory_name
    return None



def write_in_directory(directory_name, file_name, content):
    # Write a file with the specified content inside the directory
    file_path = os.path.join(directory_name, file_name)
    with open(file_path, 'w') as file:
        file.write(content)


#hyperparameter to control the data 
n_neurons = 100 #number of neurons
n_epochs = 1 #number of epoch
n_test = 2 #number of images for testing
n_train = 5 #number of images for training
exc = 22.5 #excitatory to inhibitatory weight connection
inh=120 #inhibitatory to excitatory weight connection
theta_plus=0.05 # increase of membran voltage per spike
time=250 # time of image exposition
standard_deviation = 0.0 #deviation as a percentage of mean 
nu_post =1e-2 #learning rate post
nu_pre= 1e-4 # learning rate pre

progress_interval=250
update_interval =4000


#parameter controlling how the simulation works

train = True #should train the network or use one already trained?
plot=False # should you plot everything ? Take more time to do one simulation but it's pretty.
gpu = True #should you use GPU.




params = get_STDP_param_from_data()
"""
get_STDP_param_from_data will get all the data from the directory in the path dirpath.
This means that if you have several set of data in different file you can just put all of them
"""
#params issued from the fit
tau_pres = params['tau_pre']
tau_posts = params['tau_post']
A_pres = params['A_pre']
A_posts = params['A_post']
g_mins = params['g_min']
g_maxs = params['g_max']
names = params['filenames']


# Create the directory if it doesn't exist
dirname = create_directory()

# Write the logs inside a file to save the  inside the directory





for i in range(len(tau_pres)):
    tau_pre=tau_pres[i]
    tau_post = tau_posts[i]
    A_pre= A_pres[i]
    A_post =A_posts[i]
    g_min = g_mins[i]
    g_max = g_maxs[i]
    name = names[i]
    if isnan(tau_pre) and isnan(tau_post) and isnan(A_pre) and isnan(A_post) and isnan(g_min) and isnan(g_max):
        print('error found NaN in the parameter')
        break
    print(f'get accuracy for the fit issued from the file: {name}')
    
    accuracy = getAccuracy(n_neurons = n_neurons,
                    n_epochs = n_epochs, 
                    n_test =n_test,
                    n_train = n_train, 
                    exc = exc,
                    inh=inh,
                    theta_plus=theta_plus,
                    time=time,
                    progress_interval=progress_interval,
                    update_interval =update_interval,
                    train = train,
                    plot=plot,
                    gpu = gpu,
                    tau_pre = tau_pre,
                    tau_post=tau_post,
                    A_pre = A_pre,
                    A_post = A_post,
                    g_max = g_max,
                    g_min =g_min,
                    standard_deviation =standard_deviation,
                    nu_pre = nu_pre,
                    nu_post=nu_post)
    file_content = f"Accuracy for file: {name}\n"
    file_content += f'''n_neurons = {n_neurons}\n
                    n_train = {n_train}\n
                    n_epochs = {n_epochs}\n
                    n_test = {n_test}\n
                    exc = {exc}\n
                    inh={inh}\n
                    theta_plus={theta_plus}\n
                    time={time}\n
                    progress_interval={progress_interval}\n
                    update_interval ={update_interval}\n'''
    file_content += '\n\n'
    all_activity_accuracy = accuracy["all"] / n_test
    propotion_activity_accuracy = accuracy["proportion"] / n_test
    file_content += f'All activity accuracy: {all_activity_accuracy}\n'
    file_content += f'Propotion activity accuracy : {propotion_activity_accuracy}\n'
    log_name = f'accuracy_for_file_{name}.txt'

    write_in_directory(dirname, log_name, file_content)
    
