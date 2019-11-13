import model
import numpy as onp

def array2Str(val):
    res = '[ '
    for temp in val:
        res = res + '{:<18f}'.format(temp)
    res = res + ' ]'
    return res


f = open('new_result.txt', 'r')
output = f.read()
output = output.replace('DeviceArray', 'array')

parts = []
start_idx = 0
end_idx = len(output)
temp_idx = output.find('Starting minimization at ', start_idx, end_idx)
while(temp_idx != -1):
    start_idx = temp_idx + 1
    # print(temp_idx)
    s_idx = temp_idx
    temp_idx = output.find('Starting minimization at ', start_idx, end_idx)
    # print(s_idx, temp_idx)
    parts.append(output[s_idx:temp_idx-1])

start_point = []
start_value = []
hess_inv = []
success_res = []
result_point = []

part_idx = 0
for part in parts:
    part_idx = part_idx + 1
    print('Step:', part_idx)
    # take out starting point
    temp = part[part.find('[')+1:part.find(']')-1]
    start_idx = part.find(']')+1
    temp_array = []
    for num in temp.split(' '):
        if num == '':
            continue
        temp_array.append(float(num))
    start_point.append(onp.array(temp_array))    

    # take out function value of starting point
    temp = part[part.find('fun: ')+len('fun: '):part.find('hess_inv: ')-1].replace('\n', '').replace(')', '').replace('array(', '')
    start_value.append(float(temp))

    # take out hess_inv
    temp = part[part.find('hess_inv: array(')+len('hess_inv: array('):part.find('])')+1]
    hess_part = []
    for num in temp.split('],'):
        num = num.replace('[', '').replace(']', '').replace('\n', '')
        hess_one = []
        for num_part in num.split(','):
            hess_one.append(float(num_part))
        hess_part.append(hess_one)
    hess_inv.append(onp.array(hess_part))

    # take True of False
    temp = part[part.find('success: ')+len('success: '):part.find('x: ')-1].replace('\n', '').replace(' ', '')
    success_res.append(temp)

    # take result
    temp_idx = part.find('x: array(') + len('x: array(')+1
    temp_idx1 = part.find('])', temp_idx, len(part))
    temp = part[temp_idx:temp_idx1]
    res_point = []
    for num in temp.replace(' ', '').split(','):
        res_point.append(float(num))
    result_point.append(onp.array(res_point)) 


f.close()
f = open('new_analyze.txt', 'w')
title = 'Success'.center(12) + 'Starting point'.center(70) + 'Result point'.center(80) + 'Starting Value'.center(20) + 'Result Value'.center(20)
f.write(title+'\n')

idx = 0
for point in start_point:
    result = model.model(result_point[idx], model.phif001[:50000], model.phif001[50000:5000000], model.phif021[:50000], model.phif021[50000:5000000], model.Kp[:50000],
                  model.Km[:50000], model.Pip[:50000], model.Pim[:50000], model.Kp[50000:5000000], model.Km[50000:5000000], model.Pip[50000:5000000], model.Pim[50000:5000000], model.weight_)
    hess = hess_inv[idx]
    hess_diag = []
    for i in range(hess.shape[0]):
        hess_diag.append(hess[i, i])
    # print(hess_diag)
    title = success_res[idx].center(10) + str(start_point[idx]).center(70)  + array2Str(result_point[idx]).center(80) + '    ' + str(start_value[idx]).ljust(20) + '    ' + str(result).ljust(20)
    f.write(title+'\n')
    title = ''.center(80)+ array2Str(hess_diag).center(80)
    f.write(title+'\n')
    title = ''.center(80)+ array2Str(result_point[idx]/hess_diag).center(80)
    f.write(title+'\n')
    idx = idx + 1





# print(model.model([1., 1., 1., 1.], model.phif001[:50000], model.phif001[50000:5000000], model.phif021[:50000], model.phif021[50000:5000000], model.Kp[:50000],
#                   model.Km[:50000], model.Pip[:50000], model.Pim[:50000], model.Kp[50000:5000000], model.Km[50000:5000000], model.Pip[50000:5000000], model.Pim[50000:5000000], model.weight_))

