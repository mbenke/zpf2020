import Control.Monad.Par
import Control.Monad.Fail

type In = Int
type Out = (Int, Int)

network :: IVar In -> Par Out
network inp = do
 [vf,vg,vh] <- sequence [new,new,new]

 fork $ do x <- get inp
           put vf (f x)

 fork $ do x <- get vf
           put vg (g x)

 fork $ do x <- get vf
           put vh (h x)

 x <- get vg
 y <- get vh
 return (j x y)

f x = x+1
g x = x+x
h x = x*x
j = (,)

runNetwork :: In -> Out
runNetwork n = runPar $ do
  input <- new
  put input n
  network input

main = print $ runNetwork 2

instance MonadFail Par where
  fail = error
