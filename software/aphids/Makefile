CC = gcc
CFLAGS = -fPIC -Wall
DEPS = aphids.h aphids_log.h vdif_in_databuf.h
OBJ = aphids.o aphids_log.o vdif_in_databuf.o vdif_in_null_thread.o vdif_inout_null_thread.o vdif_out_null_thread.o
DBG_ARGS = -D DEBUG

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(DBG_ARGS)

aphids.so: $(OBJ)
	$(CC) -shared -o $@ $^ -lhiredis

clean:
	rm *.o
	rm *.so
