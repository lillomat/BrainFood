# Desafio BrainFood
#### 1. ¿Qué cervecería produce la cerveza más fuerte según ABV?
<br>
1. _Cervecería que produce la cerveza especifica más fuerte_
<br><br>
Clean Nan:
    #info_beer.beer_abv = info_beer['beer_abv'].fillna(0)
<br>
Encontrar maximo ABV
    #max_abv = max(info_beer.beer_abv)
<br>


Recorrer 1.5mill y cruzar cerveceria, cerveza con el maximo encontrado
    #Empty dict
    cervecera_max_abv = {}
    for i in range(0,len(info_beer.brewery_name)):
    #Comparar con max_abv
    if max_abv == info_beer[info_beer.columns.values[11]][i]:
        cervecera_max_abv[info_beer[info_beer.columns.values[1]][i]] = info_beer[info_beer.columns.values[10]][i]
Recorrer arreglo de resultados
    #for k,v in cervecera_max_abv.items():
         print('Cervecera con mayor ABV: '+k+', Cerveza con mayor ABV: '+v)

###### Result:
Cervecera con mayor ABV: Schorschbräu
<br>
Cerveza con mayor ABV: Schorschbräu Schorschbock 57%

1. 2 _Cervecería que produce la cerveza general con mayor ABV_
<br><br>
Revisar cada empresa de cerveza
    #Empty dict
    cervecera_abv = {}
    for i in np.unique(info_beer.brewery_id):
        #Ingresar al dict, la empresa de cerveza con su promedio de ABV
        cervecera_abv[np.unique(info_beer.brewery_name[info_beer.brewery_id == i])[0]] = np.mean(info_beer.beer_abv[info_beer.brewery_id == i])
Encontrar maximo promedio de ABV
    #max_mean_abv = max(cervecera_abv.values())
Recorrer arreglo de resultados y imprimir empresa con mayor ABV prom.
    #for k,v in cervecera_abv.items():
        #Encontrar dentro del arreglo, mayor ABV promedio
        if v == max_mean_abv:
           print('Cervecera con mayor ABV prom: '+k+', promedio de ABV: '+str(v))

##### Result:
Cervecera con mayor ABV prom: Schorschbräu
<br>
Promedio de ABV: 19.2

#### 2.	¿Si tuviera que elegir 3 cervezas para recomendar usando sólo estos datos, cuáles elegiría?
