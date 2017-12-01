import urllib as ul

base_url = 'http://www.ux.uis.no/~tranden/brodatz/D'
dataset_size = 112

for i in range(1,dataset_size+1):
    ul.urlretrieve(base_url+str(i)+'.gif', './../data/brodatz/D'+str(i)+'.gif')
