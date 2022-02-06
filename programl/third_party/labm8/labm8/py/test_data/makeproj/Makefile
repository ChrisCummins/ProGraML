all: foo

foo: foo.o
	cp $< $@

clean:
	rm *.o foo

foo.o: foo.c
	cp $< $@

fail:
	false
