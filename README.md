# Image Processing GUI

### Crear el ejecutable

```
mkdir build
cd build
cmake ..
make
```

### Instrucciones:

1. Abrir directamente el programa con una imagen BMP por defecto.
```
./imgpro
```

2. Abrir una imagen en formato BMP.
```
./imgpro files/1BIT.BMP # 4BITS.BMP, 8BITS.BMP, 24BITS.BMP
```
2. Abrir una imagens como PNG, JPG, ...
```
./imgpro -t files/PNG.png
```

### Funciones
1. Adquisición y Representación de imágenes:
    - Escritura y lectura de archivos BMP de 1, 4 ,8 y 24 bits.
2. Procesamiento global de imágenes:
    - Constraste.
    - Brillo.
3. Filtros:
    - Mean Filter.
    - Laplacian Filter.
4. Transformationces Geometricas:
    - Transformaciones bilineales.
5. Espacios de color y dominio de frecuencia:
    - Escala de grises.
    - Coordenas cromáticas.
    - Transformada de Fourier
    - Transformada inversa de Fourier
6. Análisis de imágenes
    - Template matching: Con suma de diferencias al cuadrado normalizadas.


### Interface
<img src='data/readme1.png' width=900>

### Filtro Laplaciano
<img src='data/readme2.png' width=900>

### Transformada de fourier
<img src='data/readme3.png' width=900>

### Template matching
<img src='data/readme4.png' width=900>
