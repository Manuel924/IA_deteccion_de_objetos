# Uso de YOLOV3 para la deteccion de personas, gatos y plantas.

Para ejecutar en un entorno de pruebas se recomienda utilizar o modificar el archivo [app.py](/app.py), modificando la siguientes lineas de codigo a la siguiente manera:

```python
if __name__ == '__main__':
    app.run()
```

Para uso de produccion se recomienda utilizar de la siguiente manera: 

```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=[you port example 8080])
```

Recuerda que debes tener habilitado la salida de este puerto para poder ser accesible a todo el internet.

### Correr el proceso en segundo plano

Se utiliza la siguiente linea de comando en el servidor destino.

```bash
nohup python3 [name-for-file.py] &
```

Por lo que se correra en segundo plano y no corres el riesgo que se caiga tu aplicacion al cerrar la terminal.