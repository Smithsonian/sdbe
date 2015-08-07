CC = gcc
CFLAGS = -fPIC -Wall
DBG_ARGS = -D DEBUG

NVCC = nvcc
NFLAGS = -ccbin gcc --compiler-options="$(CFLAGS)" --device-c --x c

DATABUFS = vdif_in_databuf.o vdif_in_databuf_cuda.o vdif_out_databuf.o
APHIDS = aphids_db.o aphids_loop.o aphids_log.o
THREADS = vdif_in_null_thread.o vdif_inout_null_thread.o vdif_out_null_thread.o
THREADS += vdif_in_file_thread.o vdif_out_file_thread.o
#~ THREADS += vdif_inout_gpu_thread.o
OBJS = $(APHIDS) $(DATABUFS) $(THREADS)

# VDIF over network has external dependency
SGCOMM_DIR = ../sgcomm
THREADS += vdif_in_net_thread.o sgcomm_net.o

.PHONY: all clean

all: aphids.so

$(OBJS): aphids.h
$(APHIDS): %.o: %.c %.h

# External dependency for VDIF over network
sgcomm_net.o: $(SGCOMM_DIR)
	cd $(SGCOMM_DIR); \
	make sgcomm_net.o
	cp $(SGCOMM_DIR)/sgcomm_net.o .

# Need to include ../sgcomm/sgcomm_net.h
vdif_in_net_thread.o: vdif_in_net_thread.c
	$(CC) -c -o $@ $< $(CFLAGS) -I$(SGCOMM_DIR) $(DBG_ARGS)

# need CUDA compilation for vdif_in_databuf_cuda
vdif_in_databuf_cuda.o: vdif_in_databuf_cuda.cu
	$(NVCC) -c -o $@ $< $(NFLAGS) $(DBG_ARGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS) $(DBG_ARGS)

# linking should be done with CUDA-enabled compiler
aphids.so: $(OBJS)
	$(NVCC) -shared -o $@ $^ -lhiredis -lcuda -lcudart

clean:
	rm -f *.o
	rm -f *.so
