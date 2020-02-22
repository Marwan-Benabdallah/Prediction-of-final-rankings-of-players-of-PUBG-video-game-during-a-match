# %%
import itertools
import gc
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_palette('bone')

sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format


print(os.listdir("C:/Users/Marwan/Desktop/BDDAvancées"))

# %%
def product(liste1,liste2):
    return list(itertools.product(liste1,liste2))

# %%
def reduire_utilisation_mem(dataFrame):
    
    debut_mem = dataFrame.memory_usage().sum() / 1024**2
    
    for colonne in dataFrame.columns:
        
        colonne_type = dataFrame[colonne].dtype
        
        if colonne_type != object:
            
            colonne_min = dataFrame[colonne].min()
            colonne_max = dataFrame[colonne].max()
            
            if str(colonne_type)[:3] == 'int':
                
                if colonne_min > np.iinfo(np.int8).min and colonne_max < np.iinfo(np.int8).max:
                    dataFrame[colonne] = dataFrame[colonne].astype(np.int8)
                    
                elif colonne_min > np.iinfo(np.int16).min and colonne_max < np.iinfo(np.int16).max:
                    dataFrame[colonne] = dataFrame[colonne].astype(np.int16)
                    
                elif colonne_min > np.iinfo(np.int32).min and colonne_max < np.iinfo(np.int32).max:
                    dataFrame[colonne] = dataFrame[colonne].astype(np.int32)
                    
                elif colonne_min > np.iinfo(np.int64).min and colonne_max < np.iinfo(np.int64).max:
                    dataFrame[colonne] = dataFrame[colonne].astype(np.int64) 
                    
            else:
                
                if colonne_min > np.finfo(np.float32).min and colonne_max < np.finfo(np.float32).max:
                    dataFrame[colonne] = dataFrame[colonne].astype(np.float32)
                    
                else:
                    
                    dataFrame[colonne] = dataFrame[colonne].astype(np.float64)
                    
    fin_mem = dataFrame.memory_usage().sum() / 1024**2
    print("L'utilisation de la memoire du dataframe est passé de {:.2f} MB à {:.2f} MB (decroissant de {:.1f}%)".format(
        debut_mem, fin_mem, 100 * (debut_mem - fin_mem) / debut_mem))
    return dataFrame

# %%
%%time

os.chdir('C:/Users/Marwan/Desktop/BDDAvancées')

train = pd.read_csv('train_V2.csv')
train = reduire_utilisation_mem(train)

test = pd.read_csv('test_V2.csv')
test = reduire_utilisation_mem(test)

print(train.shape, test.shape)

# %%
train.info()

# %%
nbr_null = train.isnull().sum().sort_values()

print("nombre de 'null':", nbr_null[nbr_null > 0])

train.dropna(inplace=True)

# %%
train.describe(include=np.number).drop('count').T

# %%
datas = train.append(test, sort=False).reset_index(drop=True)

del train, test

gc.collect()

# %%
def remplir(dataFrame, val):
    
    nbcolonnes = dataFrame.select_dtypes(include='number').columns
    cols = nbcolonnes[nbcolonnes != 'winPlacePerc']
    
    dataFrame[dataFrame == np.Inf] = np.NaN
    dataFrame[dataFrame == np.NINF] = np.NaN
    
    for c in cols: dataFrame[c].fillna(val, inplace=True)

# %%
datas['_killPlaceOverMaxPlace'] = datas['killPlace'] / datas['maxPlace']

datas['_killsOverWalkDistance'] = datas['kills'] / datas['walkDistance']

datas['_totalDistance'] = datas['rideDistance'] + datas['walkDistance'] + datas['swimDistance']

datas['_healthItems'] = datas['heals'] + datas['boosts']

datas['_headshotKillRate'] = datas['headshotKills'] / datas['kills']

remplir(datas, 0)

# %%
match = datas.groupby('matchId')

datas['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values
datas['killsPerc'] = match['kills'].rank(pct=True).values
datas['killPlacePerc'] = match['killPlace'].rank(pct=True).values

datas['walkPerc_killsPerc'] = datas['walkDistancePerc'] / datas['killsPerc']

# %%
datas.drop(['rideDistance','swimDistance','matchDuration'], axis=1, inplace=True)
datas.drop(['rankPoints','killPoints','winPoints'], axis=1, inplace=True)

datas.drop(['boosts','heals','revives','assists'], axis=1, inplace=True)
datas.drop(['headshotKills','roadKills','vehicleDestroys','teamKills'], axis=1, inplace=True)

# %%
group = datas.groupby(['matchId','groupId','matchType'])
match = datas.groupby(['matchId'])

cols = list(datas.columns)

cols_exclues = ['Id','matchId','groupId','matchType','maxPlace','numGroups','winPlacePerc']

for c in cols_exclues:
    cols.remove(c)
print(cols)

col_somme = ['kills','killPlace','damageDealt','walkDistance','_healthItems']

# %%
data_match = pd.concat([
    match.size().to_frame('m.players'), 
    match[col_somme].sum().rename(columns=lambda s: 'm.sum.' + s), 
    match[col_somme].max().rename(columns=lambda s: 'm.max.' + s),
    match[col_somme].mean().rename(columns=lambda s: 'm.mean.' + s)
    ], axis=1).reset_index()

data_match = pd.merge(data_match, 
    group[col_somme].sum().rename(columns=lambda s: 'sum.' + s).reset_index())

data_match = reduire_utilisation_mem(data_match)

print(data_match.shape)

# %%
killsMinimum = datas.sort_values(['matchId','groupId','kills','killPlace']).groupby(
    ['matchId','groupId','kills']).first().reset_index().copy()

for n in np.arange(5):
    c = 'kills_' + str(n) + '_Place'
    nbrKills = (killsMinimum['kills'] == n)
    killsMinimum.loc[nbrKills, c] = killsMinimum[nbrKills].groupby(['matchId'])['killPlace'].rank().values
    data_match = pd.merge(data_match, killsMinimum[nbrKills][['matchId','groupId',c]], how='left')
    data_match[c].fillna(0, inplace=True)
    
data_match = reduire_utilisation_mem(data_match)
del killsMinimum, nbrKills

print(data_match.shape)

# %%
datas = pd.concat([
    group.size().to_frame('players'),
    group.mean(),
    group[cols].max().rename(columns=lambda s: 'max.' + s),
    group[cols].min().rename(columns=lambda s: 'min.' + s),
    ], axis=1).reset_index()

datas = reduire_utilisation_mem(datas)

print(datas.shape)

# %%
nbcolonnes = datas.select_dtypes(include='number').columns.values

nbcolonnes = nbcolonnes[nbcolonnes != 'winPlacePerc']

# %%
datas = pd.merge(datas, data_match)

del data_match

gc.collect()

datas['enemy.players'] = datas['m.players'] - datas['players']
for c in col_somme:
    datas['enemy.' + c] = (datas['m.sum.' + c] - datas['sum.' + c]) / datas['enemy.players']
    datas['p.max_msum.' + c] = datas['max.' + c] / datas['m.sum.' + c]
    datas['p.max_mmax.' + c] = datas['max.' + c] / datas['m.max.' + c]
    datas.drop(['m.sum.' + c, 'm.max.' + c], axis=1, inplace=True)
    
remplir(datas, 0)

print(datas.shape)

# %%
match = datas.groupby('matchId')
rangMatch = match[nbcolonnes].rank(pct=True).rename(columns=lambda s: 'rank.' + s)
datas = reduire_utilisation_mem(pd.concat([datas, rangMatch], axis=1))
col_rang = rangMatch.columns

del rangMatch

gc.collect()

match = datas.groupby('matchId')
rangMatch = match[col_rang].max().rename(columns=lambda s: 'max.' + s).reset_index()
datas = pd.merge(datas, rangMatch)

for c in nbcolonnes:
    datas['rank.' + c] = datas['rank.' + c] / datas['max.rank.' + c]
    datas.drop(['max.rank.' + c], axis=1, inplace=True)
    
del rangMatch

gc.collect()

print(datas.shape)

# %%
rangKillMinor = datas[['matchId','min.kills','max.killPlace']].copy()

group = rangKillMinor.groupby(['matchId','min.kills'])

rangKillMinor['rank.minor.maxKillPlace'] = group.rank(pct=True).values

datas = pd.merge(datas, rangKillMinor)

rangKillMinor = datas[['matchId','max.kills','min.killPlace']].copy()

group = rangKillMinor.groupby(['matchId','max.kills'])

rangKillMinor['rank.minor.minKillPlace'] = group.rank(pct=True).values

datas = pd.merge(datas, rangKillMinor)

del rangKillMinor

gc.collect()

# %%
colonne_cste = [colonne for colonne in datas.columns if datas[colonne].nunique() == 1]

print('colonnes supprimées:', colonne_cste)

datas.drop(colonne_cste, axis=1, inplace=True)

# %%
solo_duo_squad = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

datas['matchType'] = datas['matchType'].apply(solo_duo_squad)

datas = pd.concat([datas, pd.get_dummies(datas['matchType'], prefix='matchType')], axis=1)

# %%
cols = [colonne for colonne in datas.columns if colonne not in ['Id','matchId','groupId']]

for i, t in datas.loc[:, cols].dtypes.iteritems():
    if t == object:
        datas[i] = pd.factorize(datas[i])[0]

        
datas = reduire_utilisation_mem(datas)
datas.head()

# %%
X_train = datas[datas['winPlacePerc'].notnull()].reset_index(drop=True)
X_test = datas[datas['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)

del datas

gc.collect()

Y_train = X_train.pop('winPlacePerc')
X_test_groupe = X_test[['matchId','groupId']].copy()



X_train.drop(['matchId','groupId'], axis=1, inplace=True)
X_test.drop(['matchId','groupId'], axis=1, inplace=True)

X_train_colonnes = X_train.columns

print(X_train.shape, X_test.shape)

# %%
from keras.layers import PReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense

def creationModel():
    model = Sequential()
    
    model.add(Dense(512, kernel_initializer='he_normal', input_dim=X_train.shape[1], activation='relu'))
    
    model.add(BatchNormalization())
    
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_initializer='he_normal'))
    
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    
    model.add(BatchNormalization())
    
    model.add(Dropout(0.2))

    model.add(Dense(128, kernel_initializer='he_normal'))
    
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    
    model.add(BatchNormalization())
    
    model.add(Dropout(0.1))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    

    optimizer = optimizers.Adam(lr=0.005)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    
    return model

# %%
def etape_decr(lr_initial=1e-3, factor_decr=0.75, taille_etape=10, verbose=0):
    
    
    def tps(epoch):
        return lr_initial * (factor_decr ** np.floor(epoch/taille_etape))
    
    return LearningRateScheduler(tps, verbose)


lr_tps = etape_decr(lr_initial=0.001, factor_decr=0.97, taille_etape=1, verbose=1)

early_stopping = EarlyStopping(monitor='val_mean_absolute_error', mode='min', patience=10, verbose=1)

# %%
from tensorflow import set_random_seed
from sklearn import preprocessing

np.random.seed(42)
set_random_seed(1234)

scaler = preprocessing.StandardScaler().fit(X_train.astype(float))
X_train = scaler.transform(X_train.astype(float))
X_test = scaler.transform(X_test.astype(float))

model = creationModel()
history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=2**15,
        validation_split=0.2,
        callbacks=[lr_tps, early_stopping],
        verbose=2)
pred = model.predict(X_test).ravel()

# %%
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Perte du modele')

plt.xlabel('Epoch')

plt.ylabel('Loss')


plt.legend(['Train', 'Test'], loc='upper left')

plt.show()


plt.plot(history.history['mean_absolute_error'])

plt.plot(history.history['val_mean_absolute_error'])

plt.title('Erreur absolue de la moyenne')

plt.xlabel('Epoch')

plt.ylabel('Mean absolute error')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# %%
X_test = pd.read_csv('C:/Users/Marwan/Desktop/BDDAvancées/test_V2.csv', engine='python')

X_test = X_test.groupby(['matchId','groupId','matchType']).first().reset_index()

X_test = X_test[['matchId','groupId','matchType','numGroups','maxPlace','kills','killPlace']]

group = X_test_groupe.groupby(['matchId'])

X_test_groupe['winPlacePerc'] = pred

X_test_groupe['_rank.winPlacePerc'] = group['winPlacePerc'].rank(method='min')

X_test = pd.merge(X_test, X_test_groupe)

# %%
groupeComplet = (X_test['numGroups'] == X_test['maxPlace'])

ssEnsble = X_test.loc[groupeComplet]

X_test.loc[groupeComplet, 'winPlacePerc'] = (ssEnsble['_rank.winPlacePerc'].values - 1) / (ssEnsble['maxPlace'].values - 1)


ssEnsble = X_test.loc[~groupeComplet]

gap = 1.0 / (ssEnsble['maxPlace'].values - 1)

new_val = np.around(ssEnsble['winPlacePerc'].values / gap) * gap

X_test.loc[~groupeComplet, 'winPlacePerc'] = new_val

X_test['winPlacePerc'] = X_test['winPlacePerc'].clip(lower=0,upper=1)

# %%
from tqdm import tqdm

print("Vérification des anomalies dans winPlacePerc - les joueurs avec le même nombre de kills doivent avoir des scores dans l’ordre de killPlace")

correct = True
nbr_iteration = 1

while correct & (nbr_iteration <= 1000):
    X_test.sort_values(ascending=False, by=["matchId","kills","killPlace","winPlacePerc","groupId"], inplace=True)
    X_test["winPlacePerc_diff"] = X_test["winPlacePerc"].diff()
    X_test["kills_diff"] = X_test["kills"].diff()
    X_test["prev_matchId"] = X_test["matchId"].shift(1)
    X_test["prev_groupId"] = X_test["groupId"].shift(1)
    X_test["prev_winPlacePerc"] = X_test["winPlacePerc"].shift(1)
    

    dataFrame_ssEns2 = X_test[(X_test["winPlacePerc_diff"] < 0) 
                     & (X_test["kills_diff"] == 0) 
                     & (X_test["matchId"] == X_test["prev_matchId"])]
    
    nbr_anomalies = len(dataFrame_ssEns2)

    print("Iteration " + str(nbr_iteration) + " Nombre d'anomalies : " + str(nbr_anomalies))

    groupesChge = list()
    
    

    if nbr_anomalies > 0:
        print()


        dataFrame_ssEns2["new_winPlacePerc"] = dataFrame_ssEns2["winPlacePerc"] 

        dataFrame_ssEns3 = pd.DataFrame()

        
        for i in tqdm(range(1, min(15001, max(nbr_anomalies, 2))), 
                      desc="Identification des groupes uniques", mininterval=10):
            row = dataFrame_ssEns2.iloc[i - 1]
            id_precedent = str(row["prev_matchId"]) + "!" + str(row["prev_groupId"])
            id_actuel = str(row["matchId"]) + "!" + str(row["groupId"])
            if (not id_precedent in groupesChge) & (not id_actuel in groupesChge):
                groupesChge.append(id_precedent)
                groupesChge.append(id_actuel)
                dataFrame_ssEns3 = dataFrame_ssEns3.append({"matchId": row["matchId"], "groupId": row["prev_groupId"], 
                                          "new_winPlacePerc": row["winPlacePerc"]}, 
                                         sort=False, ignore_index=True)
                dataFrame_ssEns3 = dataFrame_ssEns3.append({"matchId": row["matchId"], "groupId": row["groupId"], 
                                          "new_winPlacePerc": row["prev_winPlacePerc"]}, 
                                         sort=False, ignore_index=True)

        
        dataFrame_ssEns3.drop_duplicates(inplace=True)
        
        X_test = X_test.merge(dataFrame_ssEns3, on=["matchId", "groupId"], how="left")
        
        notna = X_test["new_winPlacePerc"].notna()
        
        X_test.loc[notna, "winPlacePerc"] = X_test.loc[notna]["new_winPlacePerc"]
        
        X_test.drop(labels="new_winPlacePerc", axis=1, inplace=True)
        
        del dataFrame_ssEns2
        
        del dataFrame_ssEns3
        
        dataFrame_ssEns2 = None
        
        dataFrame_ssEns3 = None
        
        gc.collect()
    
    else:
        correct = False

    nbr_iteration = nbr_iteration + 1


if correct:
    print("Limite d'itérations atteinte")

print("Correction finie de winPlacePerc")

# %%
X_test.loc[X_test['maxPlace'] == 0, 'winPlacePerc'] = 0

X_test.loc[X_test['maxPlace'] == 1, 'winPlacePerc'] = 1 

X_test.loc[(X_test['maxPlace'] > 1) & (X_test['numGroups'] == 1), 'winPlacePerc'] = 0

X_test['winPlacePerc'].describe()

# %%
test = pd.read_csv('C:/Users/Marwan/Desktop/BDDAvancées/test_V2.csv', engine='python')

submission = pd.merge(test, X_test[['matchId','groupId','winPlacePerc']])

submission = submission[['Id','winPlacePerc']]

submission.to_csv("submission_100_Epochs.csv", index=False)