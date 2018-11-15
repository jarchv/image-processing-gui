CFLAGSLIBS = `pkg-config --cflags --libs opencv`

main:
	nvcc main.cu tools.cu -o out -std=c++11 $< $(CFLAGSLIBS) 
exec:
	./out
