CC = gcc
CFLAGS = -fPIC
DEPS = easy_databuf.h
OBJ = easy_in_thread.o easy_out_thread.o easy_databuf.o
DBG_ARGS =-D DEBUG

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(DBG_ARGS)

easy_thread.so: $(OBJ)
	$(CC) -shared -o $@ $^ 

clean:
	rm *.o
	rm easy_thread.so
