CC = gcc
CFLAGS = -fPIC -Wall
DBG_ARGS = -D DEBUG

DATABUFS = vdif_in_databuf.o
APHIDS = aphids_db.o aphids_loop.o aphids_log.o
THREADS = vdif_in_null_thread.o vdif_inout_null_thread.o vdif_out_null_thread.o
THREADS += vdif_in_file_thread.o vdif_out_file_thread.o
OBJS = $(APHIDS) $(DATABUFS) $(THREADS)

.PHONY: all clean

all: aphids.so

$(OBJS): aphids.h
$(APHIDS): %.o: %.c %.h

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS) $(DBG_ARGS)

aphids.so: $(OBJS)
	$(CC) -shared -o $@ $^ -lhiredis

clean:
	rm *.o
	rm *.so
