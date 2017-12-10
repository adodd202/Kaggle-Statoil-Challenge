    for i in tqdm(range(0, 2)):
    #for i in range(0, 51):
        models = ['resnext','senet', 'vggnet', 'densenet']


#UTILS 556
df_test_set = pd.read_json('/home/adodd202/test.json')

#UTILS 473
local_data = pd.read_json('/home/adodd202/train.json')