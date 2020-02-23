## Materiały do wykładu "Zaawansowane Programowanie Funkcyjne" Wydział MIM UW 2019/20

## "Advanced Functional Programming" course materials, MIMUW 2019/20

* Generated Lecture notes in the www subdir, source in Slides
* Generating lecture notes and slides needs pandoc

### Quick start

~~~~~
$ cabal update
$ cabal install pandoc
$ PATH=~/.cabal/bin:$PATH            # Linux
$ PATH=~/Library/Haskell/bin:$PATH   # OS X
$ git clone git://github.com/mbenke/zpf2020.git
$ cd zpf2020/Slides
$ make
~~~~~

or using stack - https://haskellstack.org/

~~~~
stack setup
stack install pandoc
export PATH=$(stack path --local-bin):$PATH
...
~~~~

On students, you can try using system GHC:

~~~~
export STACK="/home/students/inf/PUBLIC/MRJP/Stack/stack --system-ghc --resolver lts-13.19"
$STACK setup
$STACK config set system-ghc --global true
$STACK config set resolver lts-13.19
$STACK upgrade --force-download  # or cp stack executable to your path
#  ...
#  Should I try to perform the file copy using sudo? This may fail
#  Try using sudo? (y/n) n

export PATH=$($STACK path --local-bin):$PATH
$STACK install pandoc
~~~~
