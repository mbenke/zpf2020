---
title: Advanced Functional Programming
author:  Marcin Benke
date: Feb 25, 2020
---

<meta name="duration" content="80" />

# Course plan
* Types and type classes
    * Algebraic types and type classes
    * Constructor classes
    * Multiparameter classes, functional dependencies
* Testing (QuickCheck)
* Dependent types, Agda, Idris, Coq, proving properties (ca 7 weeks)
* Dependent types in Haskell
    * Type faimilies, associated tpyes, GADTs
    * data kinds, kind polymorphism
* Metaprogramming
* Parallel and concurrent programming in Haskell
    * Multicore and multiprocessor programming (SMP)
    * Concurrency
    * Data Parallel Haskell
* Project presentations

Any wishes?

# Passing the course (Zasady zaliczania)
* Lab: fixed Coq project, student-defined simple Haskell project (group projects are encouraged)
* Oral exam, most important part of which is project presentation
* Alternative to Haskell project: presentation on interesting Haskell topics during the lecture (possibly plus lab)
    * Anyone interested?

# Materials

~~~~~
$ cabal update
$ cabal install pandoc
$ PATH=~/.cabal/bin:$PATH            # Linux
$ PATH=~/Library/Haskell/bin:$PATH   # OS X
$ git clone git://github.com/mbenke/zpf2020.git && cd zpf2020
$ make
~~~~~

or using stack - https://haskellstack.org/

~~~~
stack setup
stack install pandoc
export PATH=$(stack path --local-bin):$PATH
...
~~~~

On students machine, using system GHC:

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
stack install pandoc
~~~~

# Digression - cabal and stack

**Common Architecture for Building Applications and Libraries**

`cabal install` -  lets you install libraries without root

```
[ben@students Haskell]$ cabal update
Downloading the latest package list
  from hackage.haskell.org
[ben@students Haskell]$ cabal install GLFW
...kompilacja...
Installing library in
 /home/staff/iinf/ben/.cabal/lib/GLFW-0.4.2/ghc-6.10.4
Registering GLFW-0.4.2...
Reading package info from "dist/installed-pkg-config"
 ... done.
Writing new package config file... done.
```

Many libraries on `http://hackage.haskell.org/`

# Cabal hell

```
$ cabal install criterion
Resolving dependencies...
In order, the following would be installed:
monad-par-extras-0.3.3 (reinstall) changes: mtl-2.1.2 -> 2.2.1,
transformers-0.3.0.0 -> 0.5.2.0
nats-1.1.1 (reinstall) changes: hashable-1.1.2.5 -> 1.2.5.0
...
criterion-1.1.4.0 (new package)
cabal: The following packages are likely to be broken by the reinstalls:
monad-par-0.3.4.7
void-0.7.1
lens-4.15.1
...
HTTP-4000.3.3
Use --force-reinstalls if you want to install anyway.
```

In newer cabal versions partially solved by sandboxing and `cabal new-install`

# Stack + stackage

> Stackage is a stable source of Haskell packages. We guarantee that packages build consistently and pass tests before generating nightly and Long Term Support (LTS) releases.

```
LTS 15.0 for ghc-8.8.2, published a week ago
LTS 14.27 for ghc-8.6.5, published a week ago
LTS 13.19 for ghc-8.6.4, published 10 months ago
LTS 13.11 for ghc-8.6.3, published 12 months ago
LTS 12.26 for GHC 8.4.4, published a month ago
LTS 12.14 for GHC 8.4.3, published 4 months ago
LTS 11.22 for GHC 8.2.2, published 6 months ago
LTS 9.21 for GHC 8.0.2, published a year ago
LTS 7.24 for GHC 8.0.1, published a year ago
LTS 6.35 for GHC 7.10.3, published a year ago
LTS 3.22 for GHC 7.10.2, published 3 years ago
LTS 2.22 for GHC 7.8.4, published 4 years ago
LTS 0.7 for GHC 7.8.3, published 4 years ago
```

```
$ stack --resolver lts-3.22 install criterion
Run from outside a project, using implicit global project config
Using resolver: lts-3.22 specified on command line
Downloaded lts-3.22 build plan.
mtl-2.2.1: using precompiled package
...
criterion-1.1.0.0: download
criterion-1.1.0.0: configure
criterion-1.1.0.0: build
criterion-1.1.0.0: copy/register

```

# Building a project

```
$ stack new hello --resolver lts-11.22 && cd hello
Downloading template "new-template" to create project "hello" in hello/ ...

Selected resolver: lts-11.22
Initialising configuration using resolver: lts-11.22
Total number of user packages considered: 1
Writing configuration to file: hello/stack.yaml
All done.

$ stack build
Building all executables for `hello' once. After a successful build
of all of them, only specified executables will be rebuilt.
hello-0.1.0.0: configure (lib + exe)
...
hello-0.1.0.0: copy/register
Installing library in /home/staff/iinf/ben/tmp/hello/.stack-work/install/x86_64-linux/lts-11.22/8.2.2/lib/x86_64-linux-ghc-8.2.2/hello-0.1.0.0-CaHXYhIIKYt3q9LDFmJN3m
Installing executable hello-exe in /home/staff/iinf/ben/tmp/hello/.stack-work/install/x86_64-linux/lts-11.22/8.2.2/bin
Registering library for hello-0.1.0.0..
$ stack exec hello-exe
someFunc
```

# Stack - exercises

1.  On your own machine:
    * Install `stack`
    * Install GHC 7.10 using `stack setup`
    * Install GHC 8.6 using `stack setup`
    * Run `stack ghci` with ver 7.10 and 8
    * Build and run hello project, modify it a little

2. On students you can try the same, but quota can be a problem, so use system ghc instead.


# Functional languages
* dynamically typed, strict, impure: e.g. Lisp
* statically typed, strict, impure: e.g. ML
* staticaly typed, lazy, pure: e.g. Haskell

This course: Haskell, focusing on types.

Rich type structure distinguishes Haskell among other languages.


# Types as a specification language

A function type often specifies not only its input and output but also relationship between them:


~~~~ {.haskell}
f :: forall a. a -> a
f x = ?
~~~~

If `f x` gives a result, it must be `x`

* Philip Wadler "Theorems for Free"

* `h :: a -> IO b` constructs a computation with possible side effects

    ~~~~ {.haskell}
    import Data.IORef

    f :: Int -> IO (IORef Int)
    f i = do
      print i
      r <- newIORef i
      return r

    main = do
      r <- f 42
      j <- readIORef r
      print j
    ~~~~



# Types as a specification language (2)

`g :: Integer -> Integer` may not have side effects visible outside

It may have local side effects

Example: Fibonacci numbers in constant memory

~~~~ {.haskell}
import Control.Monad.ST
import Data.STRef
fibST :: Integer -> Integer
fibST n =
    if n < 2 then n else runST fib2 where
      fib2 =  do
        x <- newSTRef 0
        y <- newSTRef 1
        fib3 n x y

      fib3 0 x _ = readSTRef x
      fib3 n x y = do
              x' <- readSTRef x
              y' <- readSTRef y
              writeSTRef x y'
              writeSTRef y (x'+y')
              fib3 (n-1) x y
~~~~

How come?

~~~~
runST :: (forall s. ST s a) -> a
~~~~

The type of `runST` guarantees that side effects do not leak;
`fibST` is pure.

# Types as a design language

* Designing programs using types and `undefined`

    ~~~~ {.haskell}
    conquer :: [Foo] -> [Bar]
    conquer fs = concatMap step fs

    step :: Foo -> [Bar]
    step = undefined
    ~~~~

Newer Haskell version allow for typed holes.

```
module Conquer where

data Foo = Foo
data Bar = Bar Foo

conquer :: [Foo] -> [Bar]
conquer fs = concatMap (pure . step) fs

step :: Foo -> Bar
step = _
```

we get

```
    • Found hole: _ :: Foo -> Bar
    • In the expression: _
      In an equation for ‘step’: step = _
    • Relevant bindings include
        step :: Foo -> Bar
      Valid substitutions include
        Bar :: Foo -> Bar
        step :: Foo -> Bar
        undefined :: forall (a :: TYPE r).
                     GHC.Stack.Types.HasCallStack =>
                     a
```

# Types as a programming language

* Functions on types computed at compile time

    ~~~~ {.haskell}
    data Zero
    data Succ n

    type One   = Succ Zero
    type Two   = Succ One
    type Three = Succ Two
    type Four  = Succ Three

    one   = undefined :: One
    two   = undefined :: Two
    three = undefined :: Three
    four  = undefined :: Four

    class Add a b c | a b -> c where
      add :: a -> b -> c
      add = undefined
    instance              Add  Zero    b  b
    instance Add a b c => Add (Succ a) b (Succ c)
    ~~~~

    ~~~~
    *Main> :t add three one
    add three one :: Succ (Succ (Succ (Succ Zero)))
    ~~~~

**Exercise:** extend with multiplication and factorial

# Types as a programming language (2)

Vectors using type classes:

~~~~ {.haskell}
data Vec :: * -> * -> * where
  VNil :: Vec Zero a
  (:>) :: a -> Vec n a -> Vec (Succ n) a

vhead :: Vec (Succ n) a -> a
vhead (x :> xs) = x
~~~~

**Exercise:** write `vtail`, `vlast`

We would like to have

~~~~ {.haskell}
vappend :: Add m n s => Vec m a -> Vec n a -> Vec s a
~~~~

but here the base type system is too weak

# Types as a programming language (3)

* Vectors with type families:

    ~~~~ {.haskell}
    data Zero = Zero
    data Suc n = Suc n

    type family m :+ n
    type instance Zero :+ n = n
    type instance (Suc m) :+ n = Suc(m:+n)

    data Vec :: * -> * -> * where
      VNil :: Vec Zero a
      (:>) :: a -> Vec n a -> Vec (Suc n) a

    vhead :: Vec (Suc n) a -> a
    vappend :: Vec m a -> Vec n a -> Vec (m:+n) a
    ~~~~


# Dependent types

Real type-level programming and proving properties is possible in a language with dependent types, such as Agda or Idris:

~~~~
module Data.Vec where
infixr 5 _∷_

data Vec (A : Set a) : ℕ → Set where
  []  : Vec A zero
  _∷_ : ∀ {n} (x : A) (xs : Vec A n) → Vec A (suc n)

_++_ : ∀ {a m n} {A : Set a} → Vec A m → Vec A n → Vec A (m + n)
[]       ++ ys = ys
(x ∷ xs) ++ ys = x ∷ (xs ++ ys)

module UsingVectorEquality {s₁ s₂} (S : Setoid s₁ s₂) where
  xs++[]=xs : ∀ {n} (xs : Vec A n) → xs ++ [] ≈ xs
  xs++[]=xs []       = []-cong
  xs++[]=xs (x ∷ xs) = SS.refl ∷-cong xs++[]=xs xs
~~~~


# A problem with dependent types

While Haskell is sometimes hard to read, dependent types are even easier to overdo:

~~~~
  now-or-never : Reflexive _∼_ →
                 ∀ {k} (x : A ⊥) →
                 ¬ ¬ ((∃ λ y → x ⇓[ other k ] y) ⊎ x ⇑[ other k ])
  now-or-never refl x = helper <$> excluded-middle
    where
    open RawMonad ¬¬-Monad

    not-now-is-never : (x : A ⊥) → (∄ λ y → x ≳ now y) → x ≳ never
    not-now-is-never (now x)   hyp with hyp (, now refl)
    ... | ()
    not-now-is-never (later x) hyp =
      later (♯ not-now-is-never (♭ x) (hyp ∘ Prod.map id laterˡ))

    helper : Dec (∃ λ y → x ≳ now y) → _
    helper (yes ≳now) = inj₁ $ Prod.map id ≳⇒ ≳now
    helper (no  ≵now) = inj₂ $ ≳⇒ $ not-now-is-never x ≵now
~~~~

...even though writing such proofs is fun.

# Parallel Haskell

Parallel Sudoku solver

~~~~ {.haskell}
main = do
    [f] <- getArgs
    grids <- fmap lines $ readFile f
    runEval (parMap solve grids) `deepseq` return ()

parMap :: (a -> b) -> [a] -> Eval [b]
parMap f [] = return []
parMap f (a:as) = do
   b <- rpar (f a)
   bs <- parMap f as
   return (b:bs)

solve :: String -> Maybe Grid
~~~~

~~~~
$ ./sudoku3b sudoku17.1000.txt +RTS -N2 -s -RTS
  TASKS: 4 (1 bound, 3 peak workers (3 total), using -N2)
  SPARKS: 1000 (1000 converted, 0 overflowed, 0 dud, 0 GC'd, 0 fizzled)

  Total   time    2.84s  (  1.49s elapsed)
  Productivity  88.9% of total user, 169.6% of total elapsed

-N8: Productivity  78.5% of total user, 569.3% of total elapsed
N16: Productivity  62.8% of total user, 833.8% of total elapsed
N32: Productivity  43.5% of total user, 1112.6% of total elapsed
~~~~

# Parallel Fibonacci

~~~~ {.haskell}
cutoff :: Int
cutoff = 20

parFib n | n < cutoff = fib n
parFib n = p `par` q `pseq` (p + q)
    where
      p = parFib $ n - 1
      q = parFib $ n - 2

fib n | n<2 = n
fib n = fib (n - 1) + fib (n - 2)
~~~~

~~~~
./parfib +RTS -N60 -s -RTS
 SPARKS: 118393 (42619 converted, 0 overflowed, 0 dud,
                 11241 GC'd, 64533 fizzled)

  Total   time   17.91s  (  0.33s elapsed)
  Productivity  98.5% of total user, 5291.5% of total elapsed

-N60, cutoff=15
  SPARKS: 974244 (164888 converted, 0 overflowed, 0 dud,
                  156448 GC'd, 652908 fizzled)
  Total   time   13.59s  (  0.28s elapsed)
  Productivity  97.6% of total user, 4746.9% of total elapsed
~~~~

# Data Parallel Haskell


~~~~ {.haskell}
{-# LANGUAGE ParallelArrays #-}
{-# OPTIONS_GHC -fvectorise #-}

module DotP where
import qualified Prelude
import Data.Array.Parallel
import Data.Array.Parallel.Prelude
import Data.Array.Parallel.Prelude.Double as D

dotp_double :: [:Double:] -> [:Double:] -> Double
dotp_double xs ys = D.sumP [:x * y | x <- xs | y <- ys:]
~~~~

Looks like list operations, but works on vectors and "automagically"
parallellises to any number of cores (also CUDA)


# Types in Haskell

* base types: `zeroInt :: Int`
* function types: `plusInt :: Int -> Int -> Int`
* polymorphic types `id :: a -> a`

    ~~~~ {.haskell}
    {-# LANGUAGE ExplicitForAll #-}
    g :: forall b.b -> b
    ~~~~

* constrained types `0 :: Num a => a`
* algebraic types

    ~~~~ {.haskell}
    data Tree a = Leaf | Node a (Tree a) (Tree a)
    ~~~~

* `Leaf`, `Node` are *value constructors

    ~~~~ {.haskell}
    data Tree a where
    	 Leaf :: Tree a
         Node :: a -> Tree a -> Tree a -> Tree a
    ~~~~

* `Tree` is a *type constructor*, an operation on types

* NB empty types are allowed:

    ~~~~ {.haskell}
    data Zero
    ~~~~

# Polymorphic typing

* Generalisation:

$${\Gamma \vdash e :: t, a \notin FV( \Gamma )}\over {\Gamma \vdash e :: \forall a.t}$$

 <!--
Jeśli $\Gamma \vdash e :: t, a \notin FV( \Gamma )$

to $\Gamma \vdash e :: \forall a.t$

  Γ ⊢ e :: t, a∉FV(Γ)
$$\Gamma \vdash e :: t$$ ,
 \(a \not\in FV(\Gamma) \) ,
to $\Gamma \vdash e :: \forall a.t$
-->

For example

$${ { \vdash map :: (a\to b) \to [a] \to [b] } \over
   { \vdash map :: \forall b. (a\to b) \to [a] \to [b] } } \over
   { \vdash map :: \forall a. \forall b. (a\to b) \to [a] \to [b] } $$

Note:

$$ f : a \to b \not \vdash map\; f :: \forall b. [a] \to [b]  $$

* Instantiation

$$ {\Gamma \vdash e :: \forall a.t}\over {\Gamma \vdash e :: t[a:=s]} $$

# Classes

* Classes describe properties of types, e.g.

    ~~~~ {.haskell}
    class Eq a where
      (==) :: a -> a -> Bool
    instance Eq Bool where
       True  == True  = True
       False == False = True
       _     == _     = False

    class Eq a => Ord a where ...
    ~~~~

* types can be cnstrained by class context:

    ~~~~ {.haskell}
    elem :: Eq a => a -> [a] -> Bool
    ~~~~

+ Implementaction
    - an instance is translated to a method dictionary (akin to C++ vtable)
    - contet is translated to an implicit parameter (method dictionary)
    - a subclass is translated to a function on method dicts


# Operations on types

* A simple example:

    ~~~~ {.haskell}
    data Tree a = Leaf | Node a (Tree a) (Tree a)
    ~~~~

* Type constructors transform types

* e.g. `Tree` maps `Int` to `Tree Int`

+ Higher order functions transform functions

+ Higher order constructors transform type constructors, e.g.

~~~~ {.haskell}
newtype IdentityT m a = IdentityT { runIdentityT :: m a }
~~~~

# Constructor classes

* constructor classes describe properties of type constructors:

    ~~~~ {.haskell}
    class Functor f where
      fmap :: (a->b) -> f a -> f b
    (<$>) = fmap

    instance Functor [] where
      fmap = map

    class Functor f => Pointed f where
       pure :: a -> f a
    instance Pointed [] where
       pure = (:[])

    class Pointed f => Applicative f where
      (<*>) :: f(a->b) -> f a -> f b

    instance Applicative [] where
      fs <*> xs = concat $ flip map fs (flip map xs)

    class Applicative m => Monad' m where
      (>>=) :: m a -> (a -> m b) -> m b
    ~~~~

<!--

    class Pointed f => Applicative f where
      (<*>) :: f(a->b) -> f a -> f b
      (*>) :: f a -> f b -> f b
      x *> y = (flip const) <$> x <*> y
      (<*) :: f a -> f b -> f a
      x <* y = const <$> x <*> y

    liftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
    liftA2 f a b = f <$> a <*> b

-->

# Kinds

* Value operations are described by their types

* Type operations are described by their kinds

* Types (e.g.. `Int`, `Int -> Bool`) are of kind `*`

* One argument constructors are of type  (e.g.. `Tree`) are of kind `* -> *`

    ~~~~ {.haskell}
    {-#LANGUAGE KindSignatures, ExplicitForAll #-}

    class Functor f => Pointed (f :: * -> *) where
        pure :: forall (a :: *).a -> f a
    ~~~~

* More complex kinds are possible, e.g. for monad transformers:

    ~~~~ {.haskell}
    class MonadTrans (t :: (* -> *) -> * -> *) where
        lift :: Monad (m :: * -> *) => forall (a :: *).m a -> t m a
    ~~~~

NB spaces are obligatory - `::*->*` is one lexem

Newer Haskell versions allow introducing user kinds - we'll talk about them later.

# Multiparameter typeclasses

* Sometimes we need to describe a relationship between types rather than just a single type:

    ~~~~ {.haskell}
    {-#LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
    class Iso a b where
      iso :: a -> b
      osi :: b -> a

    instance Iso a a where
      iso = id
      osi = id

    instance Iso ((a,b)->c) (a->b->c) where
      iso = curry
      osi = uncurry

    instance (Iso a b) => Iso [a] [b] where
     iso = map iso
     osi = map osi
    ~~~~

* NB: in the last example `iso` has a different type on the left than on the right.

* Exercise: write more instances of `Iso`, e.g.


    ~~~~ {.haskell}
    instance (Functor f, Iso a b) => Iso (f a) (f b) where
    instance Iso (a->b->c) (b->a->c) where
    ~~~~

# Digression - FlexibleInstances

Haskell 2010

<!--
An instance declaration introduces an instance of a class. Let class
cx => C u where { cbody } be a class declaration. The general form of
the corresponding instance declaration is: instance cx′ => C (T u1 …
uk) where { d } where k ≥ 0. The type (T u1 … uk) must take the form
of a type constructor T applied to simple type variables u1, … uk;
furthermore, T must not be a type synonym, and the ui must all be
distinct.
-->

* an instance head must have the form `C (T u1 ... uk)`, where `T` is a type constructor defined by a data or newtype declaration  and the `u_i` are distinct type variables

<!--
*    and each assertion in the context must have the form C' v, where v is one of the ui.
-->

This prohibits instance declarations such as:

```
  instance C (a,a) where ...
  instance C (Int,a) where ...
  instance C [[a]] where ...
```

`instance Iso a a` does not meet these conditions, but it's easy to see  what relation we mean.

# Problem with muliparameter type classes

Consider a class of collections, e.g.

`BadCollection.hs`

~~~~ {.haskell}
class Collection c where
  insert :: e -> c -> c
  member :: e -> c -> Bool

instance Collection [a] where
     insert = (:)
     member = elem
~~~~

we get an error:

~~~~
    Couldn't match type `e' with `a'
      `e' is a rigid type variable bound by
          the type signature for member :: e -> [a] -> Bool
          at BadCollection.hs:7:6
      `a' is a rigid type variable bound by
          the instance declaration
          at BadCollection.hs:5:22
~~~~

Why?

# Problem with muliparameter type classes

~~~~ {.haskell}
class Collection c where
 insert :: e -> c -> c
 member :: e -> c -> Bool
~~~~

translates more or less to

~~~~
data ColDic c = CD
 {
 , insert :: forall e.e -> c -> c
 , member :: forall e.e -> c -> Bool
 }
~~~~

 ... this is not what we meant

~~~~ {.haskell}
instance Collection [a] where
   insert = (:)
   member = undefined
~~~~

~~~~
-- (:) :: forall t. t -> [t] -> [t]
ColList :: forall a. ColDic a
ColList = \@ a -> CD { insert = (:) @ a, member = undefined }
~~~~

# Problem with muliparameter type classes

 <!--- `BadCollection2.hs` -->
<!---
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
-->

~~~~ {.haskell}
class Collection c e where
  empty :: c
  insert :: e -> c -> c
  member :: e -> c -> Bool

instance Eq a => Collection [a] a where
  empty = []
  insert  = (:)
  member = elem


ins2 x y c = insert y (insert x c)
-- ins2 :: (Collection c e, Collection c e1) => e1 -> e -> c -> c

problem1 :: [Int]
problem1 = ins2 1 2 []
-- No instances for (Collection [Int] e0, Collection [Int] e1)
-- arising from a use of `ins2'

problem2 = ins2 'a' 'b' []
-- No instance for (Collection [a0] Char)
--       arising from a use of `ins2'

problem3 :: (Collection c0 Char, Collection c0 Bool) => c0 -> c0
problem3 = ins2 True 'a'
-- Here the problem is that this is type correct, but shouldn't
~~~~


# Functional dependencies
Sometimes in multiparameter typeclasses, one parameter determines another, e.g.

~~~~ {.haskell}
 class (Monad m) => MonadState s m | m -> s where ...

 class Collects e ce | ce -> e where
      empty  :: ce
      insert :: e -> ce -> ce
      member :: e -> ce -> Bool
~~~~

Exercise: verify that `Collects` solves the problem we had with `Collection`


Problem: *Fundeps are very, very tricky.* - SPJ

More: http://research.microsoft.com/en-us/um/people/simonpj/papers/fd-chr/

# Reflection - why not constructor classes?

We could try to solve the problem this way:

~~~~ {.haskell}
class Collection c where
  insert :: e -> c e -> c e
  member :: Eq e => e -> c e-> Bool

instance Collection [] where
     insert x xs = x:xs
     member = elem
~~~~

but this does not allow to solve the problem with the state monad:

~~~~ {.haskell}
 class (Monad m) => MonadState s m | m -> s where
   get :: m s
   put :: s -> m ()
~~~~

the state type `s` is not a parameter of `m`

# Fundeps are very very tricky

~~~~ {.haskell}
class Mul a b c | a b -> c where
  (*) :: a -> b -> c

newtype Vec a = Vec [a]
instance Functor Vec where
  fmap f (Vec as) = Vec $ map f as

instance Mul a b c => Mul a (Vec b) (Vec c) where
  a * b = fmap (a*) b

f t x y = if t then  x * (Vec [y]) else y
~~~~

What is the type of `f`? Let `x::a`, `y::b`.

Then the result type of `f` is `b` and we need an instance of `Mul a (Vec b) b`

Now
Z kolei `a b -> c` implies `b ~ Vec c` for some `c`, so we are lookng for an instance

~~~~
Mul a (Vec (Vec c)) (Vec c)
~~~~

applying the rule `Mul a b c => Mul a (Vec b) (Vec c)` leads to `Mul a (Vec c) c`.

...and so on


# Let's try

~~~~ {.haskell}
Mul1.hs:16:21:
    Context reduction stack overflow; size = 21
    Use -fcontext-stack=N to increase stack size to N
      co :: c18 ~ Vec c19
      $dMul :: Mul a0 c17 c18
      $dMul :: Mul a0 c16 c17
      ...
      $dMul :: Mul a0 c1 c2
      $dMul :: Mul a0 c c1
      $dMul :: Mul a0 c0 c
      $dMul :: Mul a0 (Vec c0) c0
    When using functional dependencies to combine
      Mul a (Vec b) (Vec c),
        arising from the dependency `a b -> c'
        in the instance declaration at 3/Mul1.hs:13:10
      Mul a0 (Vec c18) c18,
        arising from a use of `mul' at 3/Mul1.hs:16:21-23
    In the expression: mul x (Vec [y])
    In the expression: if b then mul x (Vec [y]) else y
~~~~

(we need to use UndecidableInstances, to make GHC try - this example shows what is 'Undecidable').

# Type families

Type families are functions on types

~~~~ {.haskell}
{-# TypeFamilies #-}

data Zero = Zero
data Suc n = Suc n

type family m :+ n
type instance Zero :+ n = n
type instance (Suc m) :+ n = Suc(m:+n)

vhead :: Vec (Suc n) a -> a
vappend :: Vec m a -> Vec n a -> Vec (m:+n) a
~~~~

We'll talk about them systematically when we talk about dependent types in Haskell
