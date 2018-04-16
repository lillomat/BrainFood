#### 2.	¿Si tuviera que elegir 3 cervezas para recomendar usando sólo estos datos, cuáles elegiría?

##Metodo 1: usando promedio review_overall como indicador clave,
## tambien considerar un minimo de opiniones para no ser bias (ej: 2/2 opiniones buenas)
#Empty dict
quantity_opinion_beer = {}
#Conseguir todas las opiniones per beerid: {Beerid : Cantidad_Opiniones}
for i in np.unique(info_beer.beer_beerid):
    quantity_opinion_beer[i] = len(info_beer.review_overall[info_beer.beer_beerid == i])
#Ver composicion de la informacion
moda = spy.mode(list(quantity_opinion_beer.values()))#Moda=1, count= 23740
max_opinions = max(list(quantity_opinion_beer.values()))#3290
conf_interval = spy.t.interval(0.95,max_opinions, loc = np.mean(list(quantity_opinion_beer.values())),
               scale=spy.sem(list(quantity_opinion_beer.values())))
dev = np.std(list(quantity_opinion_beer.values()))
p = 0.5#Porcion conservativa
#Calcular sample size segun maximo de opiniones per beer
ss = ((1.96**2)*p*(1-p))/(0.05**2)
sample_limit = ss/(1+((ss-1)/max_opinions))
#Empty dict
beer_opinion_quality = {}
#Normalizar columna review_overall
df_norm = (info_beer.review_overall - info_beer.review_overall.mean()) / (info_beer.review_overall.max() - info_beer.review_overall.min())
#Crea dict {'Beerid':{promedio_overall}} y escoge solo aquellos sobre sample limit para eliminar overfitting
for i in np.unique(info_beer.beer_beerid):
    if quantity_opinion_beer.get(i,i) > int(sample_limit):
        r_o = np.mean(df_norm[info_beer.beer_beerid == i])
        beer_opinion_quality[i] = r_o

#Recorrer mejores 3 cervezas segun promedio de review_overall
for k,v in beer_opinion_quality.items():
    if v in sorted(beer_opinion_quality.values(), reverse=True)[:3]:
        print('Recomiendo la cerveza: '+np.unique(info_beer.beer_name[info_beer.beer_beerid == k])+',     cervecera = '+np.unique(info_beer.brewery_name[info_beer.beer_beerid == k]))

#Resultado print
# ['Recomiendo la cerveza: Trappist Westvleteren 12,     cervecera = Brouwerij Westvleteren (Sint-Sixtusabdij van Westvleteren)']
# ['Recomiendo la cerveza: Heady Topper,     cervecera = The Alchemist']
# ['Recomiendo la cerveza: Pliny The Younger,     cervecera = Russian River Brewing Company']

### Se concluye que el resultado anterior ^^^ entrega las cervezas con mayor review overall promedio per beerid,
### siempre y cuando existe una cantidad determinada de opiniones: eliminando resultados bias,ej: 5 reviews 5/5.
### Para calcular este limite de opiniones se uso el sample size: Usando como poblacion el maximo numero de
### opiniones per beerid y utilizando una porcion del 50%, que conforma un numero conservativo, (mayor tamaño muestra),
### se comparo con una porcion del 20%, en donde cambia solo una de las cervezas escogidas por otra con menor cantidad de opiniones. (248)

#%%
# 3.¿Cuál de los factores (aroma, taste, apperance, palette) es más importante
# para determinar la calidad general de una cerveza?

#Creando train array con toda la información de ('aroma','appearance','paleta','taste')
train_features = np.column_stack((info_beer.review_aroma,info_beer.review_appearance,info_beer.review_palate,info_beer.review_taste))
#train_features = train_features_full[int(len(train_features_full)*.1) : int(len(train_features_full)*0.9)]

#Creando target array con toda la información de ('overall')
train_target = np.array(info_beer.review_overall)
#train_target = train_target_full[int(len(train_target_full)*.1) : int(len(train_target_full)*0.9)]

#Usando RandomForestRegressor
#Se crean 200 arboles con criterio:mse (mean squared error) y variar max_depth y max_leaf_nodes para eliminar overfitting
rlf = skl.RandomForestRegressor(n_estimators=200,criterion='mse', random_state=0)
#Se entrenan los arboles con toda la información
rlf = rlf.fit(train_features, train_target)
#Obtiene accuracy del modelo "Final RFDT"
score = rlf.score(train_features, train_target)
"Mean accuracy of Random Forest: {0}".format(score)
#Ranking de importance del "Final RFDT"
for name, importance in zip(['Aroma','Appearance','Palate','Taste'], rlf.feature_importances_):
    print(name, "=", importance)

# Aroma = 0.004775586540899108
# Appearance = 0.005654712041229695
# Palate = 0.05128402915841135
# Taste = 0.9382856722594596 ##Variable de mayor importancia
# Mean_accuracy = 0.6739851450523608

#Usando ExtraTreesRegressor (menor tiempo procesamiento, mas bias)
#Crear 200 arboles con criterio:mse y variar max_depth y max_leaf_nodes para menor overfitting
elf = skl.ExtraTreesRegressor(n_estimators=200,random_state=0)
#Se entrenan los arboles con toda la información
elf.fit(train_features,train_target)
score_elf = elf.score(train_features, train_target)
"Mean accuracy of ExtraTrees: {0}".format(score_elf)
for name, importance in zip(['Aroma','Appearance','Palate','Taste'], elf.feature_importances_):
    print(name, "=", importance)

# Aroma = 0.06070789098689398
# Appearance = 0.011758739440004799
# Palate = 0.2308317020789784
# Taste = 0.696701667494122 ##Variable de mayor importancia
# Mean_accuracy = 0.6754159926680348

#Usando DecisionTreeRegressor (menor tiempo procesamiento, arbol simple)
#Se crea arbol de decision unico
dlf = dtl.DecisionTreeRegressor(random_state=0)
#Se entrena con la informacion
dlf.fit(train_features,train_target)
score_dlf = dlf.score(train_features,train_target)
"Mean accuracy of DecisonTrees: {0}".format(score_dlf)
for name, importance in zip(['Aroma','Appearance','Palate','Taste'], dlf.feature_importances_):
    print(name, "=", importance)

# Aroma = 0.004908861538578507
# Appearance = 0.0059185944555729805
# Palate = 0.05137172830583377
# Taste = 0.9378008157000148 ##Variable de mayor importancia
# Mean_accuracy =  0.6754159926680348

# Usando Recursive Feature elimination (tiempo de procesamiento casi nulo
# , solo indica factor con menor varianza y por ende menos importante)
#Se prueba con RegresionLineal, Lasso, HuberRegression, resultado es el mismo
model = sklm.LassoCV() #Modelo Lasso con CrossValidation
#Se pide que descarte todas las variables menos 1
rfe = sfs.RFE(model,1)
#Se entrena la regresion lineal
rfe = rfe.fit(train_features,train_target)
print(rfe.support_)
print(rfe.ranking_)
score_rfe = rfe.score(train_features,train_target)
"Mean accuracy of RFE: {0}".format(score_rfe)

#Ranking:
#Taste:1 ##Variable de mayor importancia
#Palate:2
#Aroma:3
#Appearance:4


###Se concluye que el factor de mayor importancia corresponde al "Taste" (menor mse).
###Esto se debe principalmente a la varianza entropia (medido con mse) que tiene para poder discriminar
###el valor del review_overall, en donde las otras variables como "Aroma" y "Appearance"
###no influyen en gran manera para la prediccion del review_overall.
###Como se tenian variables continuas se realizaron multiples regresiones con los 4 factores para predecir,
###el valor del review_overall, el factor con el mayor coeficiente y menor MSE corresponde a "Taste", quedando
###como mejor variable para predecir la calidad general de la cerveza

#%%

# 4.	¿Si yo típicamente disfruto una cerveza debido a su aroma y apariencia, qué estilo de cerveza debería probar?

##Usando el metodo creado en pregunta 2:
#Empty dict
quantity_opinion_beerstyle = {}

#Conseguir todas las opiniones per beer: {Beerstyle : Cantidad_Opiniones}
for i in np.unique(info_beer.beer_style):
    quantity_opinion_beerstyle[i] = len(info_beer[info_beer.beer_style == i])

#Ver composicion de la informacion
moda_style = spy.mode(list(quantity_opinion_beerstyle.values()))# No existe todos diferentes
max_opinions_style = max(list(quantity_opinion_beerstyle.values()))#117586
conf_interval_style = spy.t.interval(0.95,max_opinions_style, loc = np.mean(list(quantity_opinion_beerstyle.values())),
               scale=spy.sem(list(quantity_opinion_beerstyle.values())))
dev_style = np.std(list(quantity_opinion_beerstyle.values()))
p = 0.5#Porcion conservativa
#Calcular sample size segun maximo de opiniones per beerstyle
ss_style = ((1.96**2)*p*(1-p))/(0.05**2)
sample_limit_style = ss_style/(1+((ss_style-1)/max_opinions_style))
#Empty dict
beer_opinion_style = {}

#Crea dict {'Beerid':{promedio appearance y aroma}} y escoge solo aquellos sobre sample limit
for i in np.unique(info_beer.beer_style):
    if quantity_opinion_beerstyle.get(i,i) > int(sample_limit_style):
        r_a = np.mean(info_beer.review_aroma[info_beer.beer_style == i])
        r_ap = np.mean(info_beer.review_appearance[info_beer.beer_style == i])
        r_p = (r_a + r_ap)/2
        beer_opinion_style[i] = r_p
#Escoge el beerstyle con mayor promedio de aroma y appearance
for k,v in beer_opinion_style.items():
    if v in sorted(beer_opinion_style.values(), reverse=True)[:1]:
        print('Recomiendo beer style: '+k)
        maxprom = v
#Resultado print
#Recomiendo beer style: American Double / Imperial Stout

## Se concluye que el beerstyle American Double / Imperial Stout, compone el Estilo
## con mayor promedio de aroma y appearance de todos los otros estilos,
## se utilizo el mismo metodo de la pregunta 2 pero con diferentes factores.
## Se obtiene las opiniones por beer_style independiente del beer_id, luego se
## calcula el promedio de los factores appearance y aroma de cada beer_style,
## finalmente se promedian ambos factores para conseguir el beer_style con mayor,
## nivel de dichos factores.
## Adicionalmente se calculo mediante Decision Tree una prediccion clasificativa del beer_style,

#Codigo DT
#Creando train array con toda la información de ('aroma','appearance','paleta','taste')
train_features = np.column_stack((info_beer.review_aroma,info_beer.review_appearance))
#train_features = train_features_full[int(len(train_features_full)*.1) : int(len(train_features_full)*0.9)]

#Creando target array con toda la información de ('overall')
train_target = np.array(info_beer.beer_style)
#train_target = train_target_full[int(len(train_target_full)*.1) : int(len(train_target_full)*0.9)]

#Se crea arbol de decision unico
dlf = dtl.DecisionTreeClassifier(random_state=1)
#Se entrena con la informacion
dlf.fit(train_features,train_target)
score_dlf = dlf.score(train_features,train_target)
"Mean accuracy of DecisonTrees: {0}".format(score_dlf)
#accuracy:0.09
#Predecir Estado con Appearance:5 y Aroma:5
estilo_pred = dlf.predict([[5.,5.]])[0]
print('Modelo Recomienda: '+estilo_pred)
#Resultado print: Modelo Recomienda: American Double / Imperial Stout
r_a = np.mean(info_beer.review_aroma[info_beer.beer_style == estilo_pred])
r_ap = np.mean(info_beer.review_appearance[info_beer.beer_style == estilo_pred])
RatingDT = (r_a+r_ap)/2
RatingM1 = maxprom

##En ambos modelos se llega al mismo resultado: Estilo de Cerveza: American Double / Imperial Stout
##Pero el DT tiene un acuracy de tan solo 0.0906, esto se debe principalmente a la gran cantidad de
##posibles classes y en donde los datos pueden variar muy poco
##Se intento realizar un SVM y una red neuronal clasificativa pero el run-time era muy grande.
#%%
