BINFILES=parfib badfib sudoku1 sudoku2 sudoku3b
GHCOPTS=-O2 -rtsopts -threaded -eventlog

%: %.hs
	ghc $(GHCOPTS) -o $@ --make $^

all: $(BINFILES)

clean:
	-rm -f *.hi *.o *~

veryclean: clean
	-rm -f $(BINFILES)

.PHONY: clean veryclean
