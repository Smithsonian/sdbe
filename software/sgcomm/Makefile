CC=gcc
OPTIM=-O3
CFLAGS=-std=gnu99 -fPIC $(OPTIM) -g
LIBS=-lrt -lpthread -lscatgat
LFLAGS=$(LIBS)
DEPS=sgcomm_report.o sgcomm_net.o sgcomm_threads.o
EXEC_TRANSCEIVER=sgtxrx
OBJS_TRANSCEIVER=sgcomm_tx_rx_test.o $(DEPS)

EXEC_TRANSMITTER=sgtx
OBJS_TRANSMITTER=sgcomm_tx_test.o $(DEPS)

EXEC_RECEIVER=sgrx
OBJS_RECEIVER=sgcomm_rx_test.o $(DEPS)

$(EXEC_TRANSCEIVER): $(OBJS_TRANSCEIVER)
	$(CC) -o $@ $^ $(LFLAGS)

$(EXEC_TRANSMITTER): $(OBJS_TRANSMITTER)
	$(CC) -o $@ $^ $(LFLAGS)

$(EXEC_RECEIVER): $(OBJS_RECEIVER)
	$(CC) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY : all
all : $(EXEC_TRANSCEIVER) $(EXEC_RECEIVER) $(EXEC_TRANSMITTER)

.PHONY: clean
clean:
	rm -f *.o $(EXEC_TRANSCEIVER) $(EXEC_TRANSMITTER) $(EXEC_RECEIVER)
