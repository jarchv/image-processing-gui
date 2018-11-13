CFLAGSLIBS = `pkg-config --cflags --libs opencv`

main:
	g++ main.cpp -o out -std=c++11 $< $(CFLAGSLIBS) 
exec:
	./out
