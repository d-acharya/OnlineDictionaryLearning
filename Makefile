CC = gcc-5
CFLAGS= -O3
DEPS = mathOperations.h utilities.h
OBJ = mathOperations.o utilities.o dictionaryUpdate.o 

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

dictionaryUpdate: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f *.o *~
