CFLAGSLIBS = `pkg-config --cflags --libs opencv` 
wxFLAGS    = `wx-config --cxxflags --libs std`

main:
		g++ main.cpp tools.cpp -o out -std=c++11 $(CFLAGSLIBS) $(wxFLAGS)

cuda:
		nvcc main.cu tools.cu -o cudaout $(CFLAGSLIBS) -std=c++11
