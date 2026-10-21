CXX        = nvcc -O3 
#-arch=sm_60

SRC = radix.cu
EXECUTABLE = radix

default: compile run

.cu.o: $(SRC)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE)
	./$(EXECUTABLE) 16 2 2 4

clean:
	rm -f $(EXECUTABLE)

