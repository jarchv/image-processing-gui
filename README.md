# Image Processing GUI

Graphic interface for image processing:
1. Read and Save BMP files.
2. Filters:
    - Mean Filter
    - Laplacian Filter
```
mkdir build
cd build
cmake ..
make
./imgpro files/LAND.BMP
```

### Instructions:

1. Open directly with a default BMP image.
```
./imgpro
```

2. Open a BMP image.
```
./imgpro files/1BIT.BMP # 4BITS.BMP, 8BITS.BMP, 24BITS.BMP
```