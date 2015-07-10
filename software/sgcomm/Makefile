CC=gcc
CFLAGS=-std=gnu99
LFLAGS=-lrt -lpthread -lscatgat
DEPS=sgcomm_report.o sgcomm_net.o sgcomm_threads.o
EXEC_TRANSCEIVER=sgtxrx
OBJS_TRANSCEIVER=sgcomm_tx_rx_test.o $(DEPS)

EXEC_TRANSMITTER=sgtx
OBJS_TRANSMITTER=sgcomm_tx_test.o $(DEPS)

$(EXEC_TRANSCEIVER): $(OBJS_TRANSCEIVER)
	$(CC) -o $@ $^ $(LFLAGS)

$(EXEC_TRANSMITTER): $(OBJS_TRANSMITTER)
	$(CC) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f *.o $(EXEC_TRANSCEIVER)
