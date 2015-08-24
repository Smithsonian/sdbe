
CC = nvcc
LIBS = -lcufft
OBJS = main.o
INCLUDE =  -I/home/krosenfe/sdbe/software/aphids/

run : $(OBJS)
	$(CC) $^ -o $@ $(LIBS)

%.o : %.cu
	$(CC) $^ -c -o $@ $(INCLUDE)

