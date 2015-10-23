CC = gcc
OPTIM=-O3
CFLAGS=$(OPTIM) $(INC)
INC = -I/usr/local/cuda-7.0/targets/x86_64-linux/include/
SGCOMM_DIR = ../../sgcomm
DEPS = sgcomm_net.o
APHIDS_DIR = ../../aphids

EXEC_TRANSMITTER=transmit
EXEC_RECEIVER=receive
OBJS_TRANSMITTER=transmit.o $(DEPS)
OBJS_RECEIVER=receive.o $(DEPS)

%.o: %.c
	$(CC) -c -o $@ $^ $(CFLAGS) -I$(SGCOMM_DIR) -I$(APHIDS_DIR)

.PHONY : all
all : $(EXEC_RECEIVER) $(EXEC_TRANSMITTER)

# External dependency for VDIF over network
sgcomm_net.o: $(SGCOMM_DIR)
	cd $(SGCOMM_DIR); \
	make sgcomm_net.o
	cp $(SGCOMM_DIR)/sgcomm_net.o .

$(EXEC_TRANSMITTER) : $(OBJS_TRANSMITTER)
	$(CC) -o $@ $^ $(LFLAGS)

$(EXEC_RECEIVER) : $(OBJS_RECEIVER)
	$(CC) -o $@ $^ $(LFLAGS)
