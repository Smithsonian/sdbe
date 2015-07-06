CC=gcc
CFLAGS=-std=gnu99
LFLAGS=-lrt -lpthread -lscatgat

EXEC_TRANSCEIVER=sgtxrx
OBJS_TRANSCEIVER=sgcomm_tx_rx_test.o sgcomm_report.o sgcomm_net.o sgcomm_threads.o

$(EXEC_TRANSCEIVER): $(OBJS_TRANSCEIVER)
	$(CC) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f *.o $(EXEC_TRANSCEIVER)
