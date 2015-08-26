ifdef NET_HAMSTER
NET_HAMSTER_ = -D NET_HAMSTER
endif
ifdef GPU_MULTI
GPU_MULTI_ = -D GPU_MULTI
endif
ifdef DEBUG_DATADUMP
DEBUG_DATADUMP_ = -D DEBUG_DATADUMP=$(DEBUG_DATADUMP)
endif
DBG_ARGS = -D DEBUG $(DEBUG_DATADUMP_)

CC = gcc
CPREP = $(NET_HAMSTER_) $(GPU_MULTI_)
INC = -I/usr/local/cuda-7.0/targets/x86_64-linux/include/
CFLAGS = -fPIC -Wall -std=gnu99 $(INC) $(CPREP)

NVCC = nvcc
NFLAGS = -ccbin gcc --compiler-options="$(CFLAGS)" --device-c 

LIBS = -lcufft -lhiredis -lcuda -lcudart

DATABUFS = vdif_in_databuf.o vdif_in_databuf_cuda.o vdif_out_databuf.o vdif_out_databuf_cuda.o
APHIDS = aphids_db.o aphids_loop.o aphids_log.o
THREADS = vdif_in_null_thread.o vdif_inout_null_thread.o vdif_out_null_thread.o
THREADS += vdif_in_file_thread.o vdif_out_file_thread.o
THREADS += vdif_inout_gpu_thread.o
THREADS += vdif_in_net_thread.o vdif_out_net_thread.o

# external dependencies
SGCOMM_DIR = ../sgcomm
DEPS = sgcomm_net.o

# all objects
OBJS = $(APHIDS) $(DATABUFS) $(THREADS) $(DEPS)

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

# Need to include ../sgcomm/sgcomm_net.h
vdif_out_net_thread.o: vdif_out_net_thread.c
	$(CC) -c -o $@ $< $(CFLAGS) -I$(SGCOMM_DIR) $(DBG_ARGS)

# need CUDA compilation for vdif_in_databuf_cuda
vdif_in_databuf_cuda.o: vdif_in_databuf_cuda.cu
	$(NVCC) -c -o $@ $< $(NFLAGS) --x c $(DBG_ARGS)

# need CUDA compilation for vdif_out_databuf_cuda
vdif_out_databuf_cuda.o: vdif_out_databuf_cuda.cu
	$(NVCC) -c -o $@ $< $(NFLAGS) --x c $(DBG_ARGS)

# gpu thread
vdif_inout_gpu_thread.o: vdif_inout_gpu_thread.cu 
	$(NVCC) -c -o $@ $< $(NFLAGS) $(DBG_ARGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS) $(DBG_ARGS)

# linking should be done with CUDA-enabled compiler
aphids.so: $(OBJS)
	$(NVCC) -shared -o $@ $^ $(LIBS)

clean:
	rm -f *.o
	rm -f *.so
