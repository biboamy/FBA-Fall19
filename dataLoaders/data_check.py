import dill

BAND = ['middle']
INSTRUMENT = ['Alto Saxophone', 'Bb Clarinet', 'Flute']
SEGMENT = 2
YEAR = ['2013', '2014', '2015', '2016', '2017', '2018']

dillFile = '/media/SSD/FBA/saved_dill/middle_2_new_dataPC.dill'

performances = dill.load(open(dillFile, 'rb'))

cnt = {'2013':0, '2014':0, '2015':0, '2016':0, '2017':0, '2018':0}

for perf in performances:
    cnt[perf['year']] = cnt[perf['year']] + 1
    print(perf['year'], perf['pitch_contour'].shape)

print(cnt)