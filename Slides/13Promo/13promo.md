---
title: Advanced Functional Programming
subtitle: Pleasure and Pain of Dependent Types in Haskell
author:  Marcin Benke
date: May 25, 2020
---

<meta name="duration" content="80" />

# Plan

[Hasochism - The Pleasure and Pain of Dependently Typed Haskell Programming](http://homepages.inf.ed.ac.uk/slindley/papers/hasochism.pdf)

1. Kinds
2. GADT - [https://en.wikibooks.org/wiki/Haskell/GADT](https://en.wikibooks.org/wiki/Haskell/GADT)
3. Type promotion - [https://github.com/slindley/dependent-haskell](https://github.com/slindley/dependent-haskel)
<!--
``` {.haskell}
    data Nat = Z | S Nat
    data Vec :: Nat -> * -> * where
    vhead :: Vec (S n) a -> a
```
-->

4. Type Families
``` {.haskell}
   type family (m::Nat) :+ (n::Nat) :: Nat
   vappend :: Vec m a -> Vec n a -> Vec (m :+ n) a
   ? :: Vec(m :+ n) a -> (Vec m a, Vec n a)
```

5. Dynamic dependencies, singletons

``` {.haskell}
   data Natty :: Nat -> *
   vchop :: Natty m -> Vec (m :+ n) a -> (Vec m a, Vec n a)
   ? :: Natty m -> Vec (m :+ n) a -> Vec m a
```

# Plan B

6. Static dependencies, Proxy
``` haskell
   data NP :: Nat -> * where NP :: NP n
   vtake1 :: Natty m -> NP n -> Vec (m :+ n) -> Vec m a
```

7. Kind polymorphism
``` haskell
   data Proxy :: k -> * where Proxy :: Proxy i
   vtake2 :: Natty m -> Proxy n -> Vec (m :+ n) -> Vec m a
```

8. TypeApplication, getting rid of Proxy
``` haskell
-- >>> let v = 1 :> (1 :> (1 :> V0)); two = SS(SS SZ) in vtake4 two v
-- 1 :> (1 :> V0)
vtake4 :: forall n m a. SNat m -> Vec (m :+ n) a -> Vec m a
```

9. Type equality and proofs
``` haskell
data a :~: b = (a ~ b) => Refl
plus_id_r :: SNat n -> n :+ Z :~: n
```

# Kinds

* Operations on values are described by types

* Operations on types are described by kinds

* Types (e.g. `Int`) are of kind `*`

* One argument constructors  (e.g. `Maybe`) are of kind `* -> *`

    ~~~~ {.haskell}
    class Functor f => Pointed (f :: * -> *) where
        pure :: forall (a :: *).a -> f a
    ~~~~

* There exist also more complex kinds, e.g. for monad transformers:

    ~~~~ {.haskell}
    class MonadTrans (t :: (* -> *) -> * -> *) where
        lift :: Monad (m :: * -> *) => forall (a :: *).m a -> t m a
    ~~~~

GHC has also an  internal kind `#` for unboxed types (e.g. `Int#`)

As we shall see, more kinds may be introduced.

# Expr1

Consider a simple expression evaluator:

``` {.haskell}
data Expr = I Int
          | Add Expr Expr

eval :: Expr -> Int
eval (I n)       = n
eval (Add e1 e2) = eval e1 + eval e2
```

What if we try to add `Bool`?

``` {.haskell}
data Expr = I Int
          | B Bool
          | Add Expr Expr
          | Eq  Expr Expr
```

What type should `eval` have?

# Expr2

``` {.haskell}
data Expr = I Int
          | B Bool
          | Add Expr Expr
          | Eq  Expr Expr

-- eval :: Either Int Bool ?
-- eval (Add (B True) (I 1)) = ?

eval :: Expr -> Maybe (Either Int Bool)
eval (I n)       = Just (Left n)
eval (B n)       = Just (Right n)
eval _ = undefined       -- Exercise
```

How can we make typechecker reject `eval (Add (B True) (I 1))` ?

# Expr3 - Phantom types

```  {.haskell}
data Expr a = I Int
            | B Bool
            | Add (Expr Int) (Expr Int)
            | Eq  (Expr Int) (Expr Int)


eval :: Expr a -> a
eval (I n) = n -- Error: Couldn't match expected type ‘a’ with actual type ‘Int’
```
besides `Add (B True) (I 1)` still typechecks.

The problem is that we have

``` haskell
I :: Int -> Expr a
B :: Bool -> Expr a
Add :: Expr Int -> Expr Int -> Expr a
```
but want rather

``` haskell
I :: Int -> Expr Int
B :: Bool -> Expr Bool
Add :: Expr Int -> Expr Int -> Expr Int
```



# GADTs - Generalised Abstract Data Types

``` haskell
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE GADTs #-}

data Expr a where
  I :: Int -> Expr Int
  B :: Bool -> Expr Bool
  Add :: Expr Int -> Expr Int -> Expr Int
  Eq  :: Expr Int -> Expr Int -> Expr Bool
  -- exercise: allow comparing booleans, e.g `Eq (B True) (B True)`

eval :: Expr a -> a
eval (I n)       = n
eval (B b)       = b
eval (Add e1 e2) = eval e1 + eval e2
eval (Eq  e1 e2) = eval e1 == eval e2

deriving instance Show (Expr a)
```

# Vec

Recall an example from the first lecture

``` {.haskell}
data Zero
data Succ n

data Vec :: * -> * -> * where
  VNil :: Vec Zero a
  (:>) :: a -> Vec n a -> Vec (Succ n) a

vhead :: Vec (Succ n) a -> a
vhead (x :> xs) = x
```

Let us see how far we can go with dependent vectors in Haskell

# Promotion

If we have the `Nat` datatype, types for zero and successor can be automatically generated:

``` {.haskell}
{-# LANGUAGE GADTs, DataKinds, KindSignatures #-}
data Nat :: * where
  Z :: Nat
  S :: Nat -> Nat
```

``` {.haskell}
-- This defines
-- Type Nat
-- Value constructors: Z, S

-- Promotion (lifting) to type level yields
-- kind Nat
-- type constructors: 'Z :: Nat; 'S :: Nat -> Nat
-- 's can be omitted in most cases, but...

-- data P          -- 1
-- data Prom = P   -- 2
-- type T = P      -- 1 or promoted 2?
-- quote disambiguates:
-- type T1 = P     -- 1
-- type T2 = 'P    -- promoted 2
```

# Vec with promoted Nat

``` haskell
data Nat :: * where
  Z :: Nat
  S :: Nat -> Nat

-- Nat is a kind, and so is Nat -> * -> *
infixr 6 :>
data Vec :: Nat -> * -> * where
  V0   :: Vec 'Z a
  (:>) :: a -> Vec n a -> Vec ('S n) a

deriving instance (Show a) => Show (Vec n a)

vhead :: Vec (S n) a -> a
vhead (x:>_) = x

-- What should be the type of vtail?
```

# Other promotion examples

Heterogenous lists:

``` haskell
data HList :: [*] -> * where -- [*] is a list of types
  HNil  :: HList '[]
  HCons :: a -> HList t -> HList (a ': t)

foo0 :: HList '[]
foo0 = HNil

foo1 :: HList '[Int]
foo1 = HCons 3 HNil

foo2 :: HList [Int, Bool]
foo2 = undefined  -- (easy) exercise

```

# Vector concatenation
We have seen that addition can be defined with classes:
``` haskell
class Add (a::Nat) (b::Nat) (c::Nat)  where

instance Add  Z    b  b
instance Add a b c => Add (S a) b (S c)

vappend :: (Add m n s) => Vec m a -> Vec n a -> Vec s a
vappend V0 ys = ys
```

alas...

```
error: …
    • Could not deduce: n ~ s
      from the context: m ~ 'Z
```

The constraint checker cannot infer `s = n` from `m = 0`

# Type families

Type families provide more data for the constraint checker

``` {.haskell}
type family (n :: Nat) :+ (m :: Nat) :: Nat
type instance Z :+ m = m
type instance (S n) :+ m = S (n :+ m)

vapp :: Vec m a -> Vec n a -> Vec (m :+ n) a
vapp V0 ys = ys
vapp (x:>xs) ys = x:>(vapp xs ys)
```

Now `Z :+ m` can be reduced to `m` (at compile time)

**Exercise:** define multiplication

``` {.haskell}
type family (n :: Nat) :* (m :: Nat) :: Nat
```

# Indexing

You have probably seen some variant of `Fin` in Coq/Idris

``` {.haskell}
-- atIndex :: Vec n a -> (m < n) -> a

data Fin (n::Nat) where
    FinZ :: Fin (S n)          -- zero is less than any successor
    FinS :: Fin n -> Fin (S n) -- n is less than (n+1)

atIndex :: Vec n a -> Fin n -> a
atIndex (x:>_) FinZ = x
atIndex (_:>xs) (FinS k) = atIndex xs k

-- Exercise - why not:
-- atIndex :: Vec (S n) a -> ... ?
```

# Replicate

Let's try to define a vector counterpart of `replicate :: Int -> a -> [a]`

``` {.haskell}
vreplicate :: Nat -> a -> Vec n a
vreplicate Z _ = V0   --  Expected type: Vec n a
                      --  Actual type:   Vec 'Z a
```

more precisely, we would like

``` {.haskell}
vreplicate2 :: (n::Nat) -> a -> Vec n a
```

...but `n::Nat` has no inhabitants

*Exercise:* try your own ideas for `vreplicate`

Before we implement `vreplicate` let us look at some other functions.

# vchop

We want to write a function dual to `vappend`, chopping a vector in two

``` {.haskell}
--| chop a vector in two
vchop1 :: Vec (m :+ n) a -> (Vec m a, Vec n a)
vchop1 _ _ = undefined
```

Can we at least write a test for it?

```
-- >>> vchop1 (1 :> 2 :> V0)
-- ???
```

# vchop2
We need to count to `m`. Here's an ugly solution:

``` {.haskell}
-- | Chop a vector in two, using first argument as a measure
-- >>> vchop2 (undefined :> V0) (1 :> 2 :> V0)
-- (1 :> V0,2 :> V0)

-- NB if we had `vreplicate`, we might write
-- vchop2 (vreplicate (S Z) undefined) (1 :> 2 :> V0)

vchop2 :: Vec m x -> Vec (m :+ n) a -> (Vec m a, Vec n a)
vchop2 V0 xs = (V0, xs)
vchop2 (_:>m) (x:>xs) = (x:>ys, zs) where
  (ys, zs) = vchop2 m xs
```

# Singleton
Usin a vector is an overkill, we need just its length.

But `Nat` is not precise enough, we have `Z :: Nat`, but woud like `Zero :: SNat Z`

Idea: create a representant of every element of kind Nat

``` {.haskell}
data SNat (n::Nat) where
  SZ :: SNat Z
  SS :: SNat n -> SNat (S n)
deriving instance Show(SNat n)

add :: (SNat m) -> (SNat n) -> SNat(m :+ n)
add SZ n = n
add (SS m) n = SS (add m n)
```
**Exercise:** define multiplication
``` {.haskell}
mul :: (SNat m) -> (SNat n) -> SNat(m :* n)
```

# Comparisons and another way of vector indexing

``` haskell
-- # Comparison
type family (m::Nat) :< (n::Nat) :: Bool
type instance m :< 'Z = 'False
type instance 'Z :< ('S n) = 'True
type instance ('S m) :< ('S n) = m :< n

-- nth
nth :: (m:<n) ~ 'True => SNat m -> Vec n a -> a
nth SZ (a:>_)  = a
nth (SS m') (_:>xs) = nth m' xs
```

Apart from ordinary class constraints, we may use equality constraints.

`(m:<n) ~ 'True` is ane example of such a constraint

`nth` typechecks without this constraint, but so does `nth SZ V0` which is unsafe.

With the constraint, we get:

```
> nth SZ V0
    • Couldn't match type ‘'False’ with ‘'True’
        arising from a use of ‘nth’
    • In the expression: nth SZ V0
```

# vchop3

``` {.haskell}
-- | chop a vector in two parts
-- >>> vchop (SS SZ) (Vcons 1 (Vcons 2 V0))
-- (Vcons 1 V0,Vcons 2 V0)
vchop = vchop3
vchop3 :: SNat m -> Vec(m:+n) a -> (Vec m a, Vec n a)
vchop3 SZ xs = (V0, xs)
vchop3 (SS m) (Vcons x xs) = (Vcons x ys, zs) where
  (ys,zs) = vchop3 m xs
```

# vreplicate

``` {.haskell}
-- | `vreplicate n a` is a vector of n copies of a
-- >>> vreplicate (SS SZ) 1
-- Vcons 1 V0
-- >>> vreplicate (SS (SS SZ)) 1
-- Vcons 1 (Vcons 1 V0)
vreplicate :: SNat n -> a -> Vec n a
vreplicate SZ _ = V0
vreplicate (SS n) x = Vcons x (vreplicate n x)
```

**Exercise:** define

``` {.haskell}
vcycle :: SNat n -> Vec m a -> Vec (n:*m) a
```

# vtake

We want to define a vector counterpart of `take`
``` {.haskell}
{-# LANGUAGE AllowAmbiguousTypes #-}

vtake1 :: SNat m -> Vec (m :+ n) -> Vec m x
vtake1  SZ     xs     = V0
vtake1 (SS m) (x:>xs) = x :> vtake1 m xs
```

``` {.error}
error: …
    • Could not deduce: (n1 :+ n0) ~ n2
      from the context: m ~ 'S n1
        bound by a pattern with constructor:
                   SS :: forall (n :: Nat). SNat n -> SNat ('S n),
                 in an equation for ‘vtake1’
```

The compiler cannot type the recursive case.
The problem is whether `(m :+)` is injective.

NB `AllowAmbiguousTypes` is needed if we want the compiler to even try
to typecheck this.

# Injectivity

`Maybe a ~ Maybe b => a ~ b`

but it's harder to see, if

`m :+ n0 ~ m :+ n1 => n0 ~ n1`

More precisely in the type

`vtake1 :: SNat m -> Vec (m :+ n) -> Vec m x`

we lack a "handle" on `n`; with real dependent types we would write

```
(m : Nat) -> (n : Nat) -> Vec (m + n) x -> Vec m x
```

# `Proxy`  - the handle

Let us try to build a handle:

``` haskell
-- | Nat Proxy
data NP :: Nat -> * where NP :: NP n

-- >>> let v = 1 :> (1 :> (1 :> V0)); two = SS(SS SZ) in vtake2 two NP v
-- 1 :> (1 :> V0)
vtake2 :: SNat m -> NP n -> Vec (m :+ n) a -> Vec m a
vtake2 SZ     _ _ = V0
vtake2 (SS m) n (x:>xs) = x :> vtake2 m n xs
```
Note: this is different from a singleton.

# A universal handle

``` haskell
-- | Nat Proxy
data NP :: Nat -> * where NP :: NP n
```

there is no reason why our handle should depend on `Nat`, so why not make it polymorphic?

``` haskell
{-# LANGUAGE PolyKinds #-}
-- | Generic Proxy
data Proxy :: k -> * where
  Proxy :: Proxy (i::k)

-- >>> let v = 1 :> (1 :> (1 :> V0)); two = SS(SS SZ) in vtake3 two Proxy v
-- 1 :> (1 :> V0)
vtake3 :: SNat m -> Proxy n -> Vec (m :+ n) a -> Vec m a
vtake3 SZ     _ _ = V0
vtake3 (SS m) n (x:>xs) = x :> vtake3 m n xs
```

**Uwaga:** `k` is a kind variable, hence the need for `PolyKinds`

# Back to the future or another take on vtake1

Since 8.0  GHC allows explicit type applications, e.g.

```
Prelude> :set -XTypeApplications
Prelude> :t read
read :: Read a => String -> a
Prelude> read @Int "42"
42
Prelude> read @Double "42"
42.0
```

``` haskell
-- vtake4 requires:
-- {-# LANGUAGE ScopedTypeVariables #-}
-- {-# LANGUAGE TypeApplications #-} -- GHC>=8.0

-- >>> let v = 1 :> (1 :> (1 :> V0)); two = SS(SS SZ) in vtake4 two v
-- 1 :> (1 :> V0)
vtake4 :: forall n m a. SNat m -> Vec (m :+ n) a -> Vec m a
vtake4 SZ _ = V0
vtake4 (SS m) (x:>xs) = x :> vtake4 @n m xs
```

# Type Equality and proofs

``` haskell
-- cf Data.Type.Equality

infix 4 :~:

data a :~: b where
  Refl ::  a :~: a
-- data a :~: b = (a ~ b) => Refl

sym :: (a :~: b) -> (b :~: a)
sym Refl = Refl  -- seems trivial, but see if you can simplify it...

trans :: (a :~: b) -> (b :~: c) -> (a :~: c)
trans Refl Refl = Refl

cong :: forall f a b.a :~: b -> f a :~: f b
cong Refl = Refl

-- (a ~ b) implies (f a) implies (f b)
subst :: a :~: b -> f a -> f b
subst Refl = id
```

# Some proofs

``` haskell
-- Trivial lemma: 0+n ~ n; explicit quantification
plus_id_l :: SNat n -> Z :+ n :~: n
plus_id_l _ = Refl

-- implicit quantification
plus_id_l_impl :: n :~: Z :+ n
plus_id_l_impl = Refl

-- Prove by induction: n+0 ~ n
-- Pattern-match on n, so explicit quantification
-- Compare with Coq Agda or Idris
plus_id_r :: SNat n -> n :+ Z :~: n
plus_id_r SZ = Refl
plus_id_r (SS m) = cong @S (plus_id_r m)
-- @S is optional above, added only for clarity
```

# Provably safe casts

``` haskell
-- | Typesafe cast using propositional equality
-- simple but not very useful (subst is more powerful)
castWith :: (a :~: b) -> a -> b
castWith Refl x = x

-- | Generalised form of typesafe cast
gcastWith :: (a :~: b) -> (a ~ b => r) -> r
gcastWith Refl x = x

simpl0r :: SNat n -> f (n:+Z) -> f n
-- Special case: Vec (n:+Z) a -> Vec n a
simpl0r n v = castWith (cong (plus_id_r n)) v
-- simpl0r n v = subst (plus_id_r n) v

expand0r :: SNat n -> f n -> f(n:+Z)
expand0r n x = subst (sym (plus_id_r n)) x

-- Instead of `subst ... sym` we can put the constraint solver to work
-- you can think of it as a kind of tactic
expand0r' :: SNat n -> f n -> f(n:+Z)
expand0r' n x = gcastWith (plus_id_r n) x

```

# One more lemma

``` haskell
-- n + S m ~ S(m+n)
-- implicit m, explicit n
plus_succ_r :: forall m n. SNat n -> n :+ S m :~: S(n :+ m)
plus_succ_r SZ = Refl
plus_succ_r (SS n1) = cong @S (plus_succ_r @m n1)

-- explicit m, n
plus_succ_r2 :: SNat n -> SNat m -> n :+ S m :~: S(n :+ m)
plus_succ_r2 SZ m = Refl
plus_succ_r2 (SS n1) m = cong @S (plus_succ_r2 n1 m)
```

# Type-safe reverse

``` haskell
rev :: [a] -> [a]
rev [] = []
rev xs = go [] xs where
  go acc [] = acc
  go acc (h:t) = go (h:acc) t

accrev :: Vec a n -> Vec a n
accrev V0 = V0
accrev xs = go SZ V0 xs where
  go :: forall m n a.SNat n -> Vec a n -> Vec a m -> Vec a (n:+m)
  go alen acc V0 = expand0r alen acc
  go alen acc (h:>t) = gcastWith (plus_succ_r2 alen (size t))
                     $ go (SS alen) (h:>acc) t

main = print $ accrev $ 1 :> 2 :> 3 :> V0

size :: Vec a n -> SNat n
size V0 = SZ
size (_:>t) = SS (size t)
```

# Exercises

Exercise: implement a vector variant of

``` haskell
naiverev :: [a] -> [a]
naiverev [] = []
naiverev (x:xs) = naiverev xs ++ [x]
```

Challenge: try to eliminate `size` from `accrev` by using proxies or type app
(there is `Data.Vect`, stackoverflow and blogs but try to roll your own before you peek at other solutions).

# Type level literals

``` haskell
{-# LANGUAGE KindSignatures, DataKinds #-}

import GHC.TypeLits (Symbol) -- strings promoted to types
import Data.Ratio ((%))

newtype Money (currency :: Symbol) = Money Rational deriving Show

fivePence :: Money "GBP"
fivePence = Money (5 % 100)

twoEuros :: Money "EUR"
twoEuros = Money 2

add :: Money c -> Money c -> Money c
add (Money x) (Money y) = Money (x + y)

-- >>> add fivePence fivePence
-- Money (1 % 10)

-- >>> add fivePence twoEuros
-- <interactive>:18:15: error:
--    • Couldn't match type ‘"EUR"’ with ‘"GBP"’
--      Expected type: Money "GBP"
--        Actual type: Money "EUR"
```

# Type level numbers, kind level tuples

``` haskell
{-# LANGUAGE KindSignatures, DataKinds #-}

import GHC.TypeLits (Symbol, Nat)

newtype Discrete (currency :: Symbol) (scale :: (Nat, Nat))
  = Discrete Integer

oneDollar :: Discrete "USD" '(1, 1)
oneDollar = Discrete 1

oneDollarThirtyCents :: Discrete "USD" '(100, 1)
oneDollarThirtyCents = Discrete 130
```

In `scale :: (Nat, Nat)`, `(,)` is the tuple type promoted to a kind via  DataKinds.

In `'(100, 1), '(,)` is the tuple data constructor promoted to a type constructor.

# Guest Lecture

### Jun 11, 8.30 room 3180

### Ruud Koot from Standard Chartered

will talk about Haskell in banking and finance
