import dill
import numpy as np
import os

dim = 900

PATH_FBA_SPLIT = "/media/Data/split_dill/"
PATH_FBA_DILL_OLD = '/media/Data/fall19/data/pitch_contour/'

oldsplit_dill_name = PATH_FBA_DILL_OLD + 'middle_2_pc_3_test.dill'

pc_old = dill.load(open(oldsplit_dill_name, 'rb'))

old_matrix = "/home/jhuang/checkMatrix/matrix_fixed_test{}.dill".format(dim)
new_matrix = "/media/Data/fall19/data/matrix/middle_matrix_fixed_test{}_oldsplit.dill".format(dim)

mtx_old = dill.load(open(old_matrix, 'rb'))
mtx_new = dill.load(open(new_matrix, 'rb'))

print(len(pc_old), len(mtx_old), len(mtx_new))

for i in np.arange(len(pc_old)):
    perf_pc_old = pc_old[i]
    perf_old = mtx_old[i]
    assert (perf_pc_old['ratings'] == perf_old['ratings'])
    if perf_pc_old['instrumemt'] == "Alto Saxophone":
        continue
    print("-------------------------------")
    for j in np.arange(len(mtx_new)):
        perf_new = mtx_new[j]
        if (perf_new['ratings'][:3] == perf_old['ratings'][:3]):
            print(np.max(np.abs(perf_new['matrix']-perf_old['matrix'])))

# assert (perf_old['ratings'] == perf_new['ratings'])
print(perf_old['ratings'], perf_new['ratings'])

rsz_old = perf_old['matrix']
rsz_new = perf_new['matrix']
print('')

# new_pc = "/media/Data/saved_dill/middle_2_pc_6_fix.dill"
# old_pc = "/media/Data/saved_dill/middle_2_pc_3.dill"
#
# pc_old = dill.load(open(old_pc, 'rb'))
# pc_new = dill.load(open(new_pc, 'rb'))
#
# perf_old = [(perf['year'], perf['student_id'], perf['ratings'][:3]) for perf in pc_old]
# perf_new = [(perf['year'], perf['student_id'], perf['ratings'][:3]) for perf in pc_new]
#
# for perf_n in perf_new:
#     if perf_n in perf_old:
#         continue
#     for perf_o in perf_old:
#         if perf_n[0] == perf_o[0] and perf_n[1] == perf_o[1]:
#             print(perf_n, perf_o)