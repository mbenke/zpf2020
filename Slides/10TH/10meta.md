---
title: Advanced Functional Programming
subtitle: Template Haskell
author:  Marcin Benke
date: May 4, 2020
---

# Metaprogramming - Template Haskell

Code for today is on github:

* Code/TH/Here - multiline strings with TH (aka here docs)
* Code/TH/Projections - building declarations in TH
* Code/TH/QQ  - quasiquotation

# Problem: multiline strings


``` {.haskell}
showClass :: [Method] -> String
showClass ms = "\
\.class  public Instant\n\
\.super  java/lang/Object\n\
\\n\
\;\n\
\; standard initializer\n\
\.method public <init>()V\n\
\   aload_0\n\
\   invokespecial java/lang/Object/<init>()V\n\
\   return\n\
\.end method\n" ++ unlines (map showMethod ms)
```

# Template Haskell

Multiline strings in Haskell according to Haskell Wiki:

```
{-# LANGUAGE QuasiQuotes #-}
module Main where
import Str

longString = [str|This is a multiline string.
It contains embedded newlines. And Unicode:

Ἐν ἀρχῇ ἦν ὁ Λόγος

It ends here: |]

main = putStrLn longString

```

``` {.haskell}
module Str(str) where

import Language.Haskell.TH
import Language.Haskell.TH.Quote

str = QuasiQuoter { quoteExp = stringE }
```

Let's try to understand how it works...

# Parsing Haskell code at runtime

Quotations - `[q| ... |]` are a mechanism for generating ASTs.
The quasiquoter `q` determines how the bracket content is parsed
(default is `e` for Haskell expressions).

This [tutorial](https://web.archive.org/web/20180501004533/http://www.hyperedsoftware.com:80/blog/entries/first-stab-th.html)
recommends experimenting in GHCi:

```
$ ghci -XTemplateHaskell
> :m +Language.Haskell.TH
> runQ [| \x -> 1 |]

LamE [VarP x_0] (LitE (IntegerL 1))

> :t it
it :: Exp

> :i Exp
data Exp
  = VarE Name
  | ConE Name
  | LitE Lit
...
  	-- Defined in ‘Language.Haskell.TH.Syntax’

> runQ [| \x -> x + 1 |]  >>= putStrLn . pprint
\x_0 -> x_0 GHC.Num.+ 1
```

# Q, runQ

```
> :t [| \x -> 1 |]
[| \x -> 1 |] :: ExpQ
> :i ExpQ
type ExpQ = Q Exp 	-- Defined in ‘Language.Haskell.TH.Lib.Internal’

> :i Q
newtype Q a = ... -- Defined in ‘Language.Haskell.TH.Syntax’
instance Monad Q

> :t runQ
runQ :: Language.Haskell.TH.Syntax.Quasi m => Q a -> m a

>: i Quasi
class (MonadIO m, MonadFail m) => Quasi m where ...
instance Quasi Q
instance Quasi IO
```

Basically `runQ` can be used to evaluate `Q` computations both in the `Q` context (natural habitat) and the `IO` context (useful for experimentation).

<!--
(curious about `type role Q nominal`? - see e.g. this [question](https://stackoverflow.com/questions/49209788/simplest-examples-demonstrating-the-need-for-nominal-type-role-in-haskell)
-->

# Splicing structure trees into a program

```
> runQ [| succ 1 |]
AppE (VarE GHC.Enum.succ) (LitE (IntegerL 1))

> $(return it)
2

> int = LitE . IntegerL
> $(return (int 42))
42

> 1 + $(return (int 41))
42
```

but:
```
> $(return (AppE (VarE GHC.Enum.succ) (LitE (IntegerL 1))))

<interactive>: error:
    Couldn't match expected type ‘Name’ with actual type ‘a0 -> a0’
    Probable cause: ‘succ’ is applied to too few arguments
    In the first argument of ‘VarE’, namely ‘succ’
    In the first argument of ‘AppE’, namely ‘(VarE succ)’
> $(return (AppE (VarE "GHC.Enum.succ") (LitE (IntegerL 1))))
<interactive>: error:
    • Couldn't match expected type ‘Name’ with actual type ‘[Char]’
> :t VarE
VarE :: Name -> Exp
```

`VarE` needs a `Name`

# Making a Name from String

`VarE` needs a `Name`

```
> :t VarE
VarE :: Name -> Exp

> :t mkName
mkName :: String -> Name

> $( return (AppE (VarE (mkName "succ")) (LitE (IntegerL 1))))
2

```


# Names, patterns, declarations


So far, we have been building expressions, but we can build patterns, declarations, etc.:

```
> runQ [d| p1 (a,b) = a |]
[FunD p1_0 [Clause [TupP [VarP a_1,VarP b_2]] (NormalB (VarE a_1)) []]]
```

`FunD` etc - see  [documentation](https://hackage.haskell.org/package/template-haskell-2.14.0.0/docs/Language-Haskell-TH.html#g:18).

``` {.haskell}
data Clause = Clause [Pat] Body [Dec]  -- f pats = b where decs
data Dec                               -- declaration
  = FunD Name [Clause]        
  ...
```

Let us now try to build such a definition ourselves.

Note that we need to use two modules, since definitions to be run during compilation have to be imported from a different module --- the code to be run needs to be compiled first.

Otherwise you may see an error like

```
GHC stage restriction:
      ‘build_p1’ is used in a top-level splice, quasi-quote, or annotation,
      and must be imported, not defined locally
```

# Build1


``` {.haskell}
{-# START_FILE Build1.hs #-}
{-# LANGUAGE TemplateHaskell #-}
module Build1 where
import Language.Haskell.TH

-- p1 (a,b) = a
build_p1 :: Q [Dec]
build_p1 = return
    [ FunD p1
             [ Clause [TupP [VarP a,VarP b]] (NormalB (VarE a)) []
             ]
    ] where
       p1 = mkName "p1"
       a = mkName "a"
       b = mkName "b"

{-# START_FILE Declare1.hs #-}
{-# LANGUAGE TemplateHaskell #-}
import Language.Haskell.TH
import Build1

$(build_p1)

main = print $ p1 (1,2)
```


# Printing the declarations we built

``` {.haskell}
import Build1

$(build_p1)

pprLn :: Ppr a => a -> IO ()
pprLn = putStrLn . pprint
-- pprint :: Ppr a => a -> String

main = do
  decs <- runQ build_p1
  pprLn decs
  print $ p1(1,2)
```

```
p1 (a, b) = a
1
```

Reminder about `runQ`:
``` {.haskell }
class Monad m => Quasi m where ...
instance Quasi Q where ...
instance Quasi IO where ...
runQ :: Quasi m => Q a -> m a
```

# Fresh names

Building and transforming structure trees for a language with bindings
is complicated because of possible name conflicts.

Luckily, TH provides the function
[newName](https://hackage.haskell.org/packages/archive/template-haskell/2.14.0.0/doc/html/Language-Haskell-TH.html#v:newName):

```
newName :: String -> Q Name

> runQ ((,) <$> newName "x" <*> newName "x" )
(x_1,x_2)
```

(which, by the way, explains one of the reasons why
[Q](https://hackage.haskell.org/packages/archive/template-haskell/2.14.0.0/doc/html/Language-Haskell-TH.html#t:Q) needs to be a monad).

Using `newName` we can safeguard our code against name clashes.

Note, however, that `p1` is global and must use `mkName`,
while `a` and `b` are locals, so we shall generate them using `newName`.

# Build2

``` haskell
{-# START_FILE Build2.hs #-}
{-# LANGUAGE TemplateHaskell #-}
module Build2 where
import Language.Haskell.TH

build_p1 :: Q [Dec]
build_p1 = do
  let p1 = mkName "p1"
  a <- newName "a"
  b <- newName "b"
  return
    [ FunD p1
             [ Clause [TupP [VarP a,VarP b]] (NormalB (VarE a)) []
             ]
    ]

{-# LANGUAGE TemplateHaskell #-}
import Language.Haskell.TH
import Build2

$(build_p1)

main = print $ p1 (1,2)
```

# Typical TH use

Let us define all projections for large (say 16-) tuples.
Writing this by hand is no fun, but TH helps avoid the boilerplate.

Here we start by pairs, but extending it to larger tuples is a simple exercise.

An auxiliary function building a simple declaration may come handy, e.g.

``` haskell
simpleFun name pats rhs = FunD name [Clause pats (NormalB rhs) []]
```

Given a function such that `build_p n` builds the nth projection,
we can build them all using `mapM`

``` haskell
build_ps = mapM build_p [1,2]
```

Then we may splice the definitions into the program

``` haskell
$(build_ps)

main = mapM_ print
  [ p2_1 (1,2)
  , p2_2 (1,2)
  ]
```

# Build3

``` {.haskell}
{-# START_FILE Build3.hs #-}
module Build3 where
import Language.Haskell.TH

simpleFun :: Name -> [Pat] -> Exp -> Dec
simpleFun name pats rhs = FunD name [Clause pats (NormalB rhs) []]

build_ps = mapM build_p [1,2] where
    fname n = mkName $ "p2_" ++ show n
    build_p n = do
        argNames <- mapM newName (replicate 2 "a")
        let args = map VarP argNames
        return $ simpleFun (fname n) [TupP args] (VarE (argNames !! (n-1)))

{-# START_FILE Declare3.hs #-}
{-# LANGUAGE TemplateHaskell #-}

import Build3
build_ps -- one may omit $(...) for declarations

main = mapM_ print
    [ p2_1 (1,2)
    , p2_2 (1,2)
    ]
```

```
1
2
```

# Quote, eval, quasiquote

In Lisp we have quote: `'` (`code -> data`)

```
(+ 1 1)         => 2
'(+ 1 1)        => (list '+ 1 1)
(eval '(+ 1 1)) => 2
(1 2 3)         ERROR
'(1 2 3)        => (list 1 2 3)
'(1 (+ 1 1) 3)  => (list 1 '(+ 1 1) 3)
```

and a slightly more involved quasiquote/unquote pair: `` `/, `` (backtick/comma)

```
`(1 ,(+ 1 1) 3) => (list 1 2 3)
```

enabling us to evaluate some fragments inside quoted code.

In Lisp there are only S-expressions, Haskell syntax is more complex:

* expressions
* patterns
* types
* declarations

# Quasiquoting

We have seen the standard quasiquoters e, t, d, p (e.g. `[e| \x -> x +1|]` ).
We can also define our own:

``` haskell
longString = [str|This is a multiline string.
It contains embedded newlines. And Unicode:

Ἐν ἀρχῇ ἦν ὁ Λόγος

It ends here: |]

main = putStrLn longString
```

``` {.haskell}
module Str(str) where

import Language.Haskell.TH
import Language.Haskell.TH.Quote

str = QuasiQuoter { quoteExp = stringE }
```


* `stringE` builds a string literal expression
* `str` quasiquoter, when used in expression context, splices this literal

# The QuasiQuoter type

```
> :i QuasiQuoter
data QuasiQuoter
  = QuasiQuoter {quoteExp :: String -> Q Exp,
                 quotePat :: String -> Q Pat,
                 quoteType :: String -> Q Type,
                 quoteDec :: String -> Q [Dec]}
  	-- Defined in ‘Language.Haskell.TH.Quote’
```

```haskell
str = QuasiQuoter { quoteExp = stringE }
```

We intend to use `str` only in expression contexts,
so we leave the other parts undefined.

# Parsing Expressions

Let's start with a simple data type and parser for arithmetic expressions


``` { .haskell }
{-# LANGUAGE DeriveDataTypeable #-}

data Expr = EInt Int
  | EAdd Expr Expr
  | ESub Expr Expr
  | EMul Expr Expr
  | EDiv Expr Expr
    deriving(Show,Typeable,Data)
-- deriving Data needed to use generic function
-- liftData :: Data a => a -> ExpQ

pExpr :: Parser Expr
-- ...

test1 = parse pExpr "test1" "1 - 2 - 3 * 4 "
main = print test1
```

```
Right (ESub (ESub (EInt 1) (EInt 2)) (EMul (EInt 3) (EInt 4)))
```

# Building test cases

Now let's say we need some expresion trees in our program. For this kind of expressions we could (almost) get by  with `class Num` hack:

``` { .haskell }
instance Num Expr where
  fromInteger = EInt . fromInteger
  (+) = EAdd
  (*) = EMul
  (-) = ESub

testExpr :: Expr
testExpr = (2 + 2) * 3
```

...but it is neither extensible nor, in fact, nice.

# Building test cases via parsing

Of course as soon as we have a parser ready we could use it to build expressions

``` { .haskell }
testExpr = parse pExpr "testExpr" "1+2*3"
```
...but then potential errors in the expression texts remain undetected until runtime, and also this is not flexible enough: what if we wanted a simplifier for expressions, along the lines of

``` { .haskell }
simpl :: Expr -> Expr
simpl (EAdd (EInt 0) x) = x
```

# Why it's good to be Quasiquoted

what if we could instead write

``` { .haskell }
simpl :: Expr -> Expr
simpl (0 + x) = x
```

turns out with quasiquotation we can do just that (albeit with a slightly different syntax), so to whet your appetite:

``` { .haskell }
simpl :: Expr -> Expr
simpl [expr|0 + $x|] = x

main = print $ simpl [expr|0+2|]
-- ...
expr  :: QuasiQuoter
expr  =  QuasiQuoter
  { quoteExp = quoteExprExp
  , quotePat = quoteExprPat
  }
```

 Let us start with the (perhaps simplest) quasiquoter for expressions:


``` { .haskell }
quoteExprExp :: String -> Q Exp
quoteExprExp s = do
  pos <- getPosition
  exp <- parseExp pos s
  exprToExpQ exp
```

# Quasiquoting Expressions

There are three steps:

* record the current position in Haskell file (for parse error reporting);
* parse the expression into our abstract syntax;
* convert our abstract syntax to its Template Haskell representation.

The first step is accomplished using [Language.Haskell.TH.location](http://hackage.haskell.org/packages/archive/template-haskell/2.14.0.0/doc/html/Language-Haskell-TH.html#v:location) and converting it to something usable by Parsec:

``` haskell
getPosition = fmap transPos location where
  transPos loc = (loc_filename loc,
                  fst (loc_start loc),
                  snd (loc_start loc))
```

Parsing is done with our expression parser, but building the Haskell AST is a bit of work.

# Building AST
Next we need to build Haskell AST from expression tree built by our parser:

``` haskell
exprToExpQ :: Expr -> Q Exp
exprToExpQ (EInt n) = return $ ConE (mkName "EInt") $$ (intLitE n)
exprToExpQ (EAdd e1 e2) = convertBinE "EAdd" e1 e2
exprToExpQ (ESub e1 e2) = convertBinE "ESub" e1 e2
exprToExpQ (EMul e1 e2) = convertBinE "EMul" e1 e2
exprToExpQ (EDiv e1 e2) = convertBinE "EDiv" e1 e2

convertBinE s e1 e2 = do
  e1' <- exprToExpQ e1
  e2' <- exprToExpQ e2  
  return $ ConE (mkName s) $$ e1' $$ e2'
```

(alternatively we might make our parser return Haskell AST)

# Scrap Your Boilerplate

This seems like a lot of boilerplate,
luckily we can save us some work use facilities for generic programming provided by
[Data.Data](http://hackage.haskell.org/package/base/docs/Data-Data.html)
combined with the Template Haskell function
[dataToExpQ](http://hackage.haskell.org/package/template-haskell-2.14.0.0/docs/Language-Haskell-TH-Syntax.html#v:dataToExpQ),

``` haskell
 exprToExpQ =  dataToExpQ (const Nothing) exp
-- dataToExpQ :: Data a
--            => (forall b. Data b => b -> Maybe (Q Exp))
--            -> a -> Q Exp
-- the first argument provides a way of extending the translation
```
or a simpler [liftData](http://hackage.haskell.org/package/template-haskell-2.14.0.0/docs/Language-Haskell-TH-Syntax.html#v:liftData)

``` haskell
liftData :: Data a => a -> Q Exp
```

# Quasiquoting patterns

So far, we are halfway through to our goal: we can use the quasiquoter
on the right hand side of function definitions:

``` { .haskell }
testExpr :: Expr
testExpr = [expr|1+2*3|]
```

To be able to write things like

``` { .haskell }
simpl [expr|0 + $x|] = x
```
we need to write a quasiquoter for patterns.

# Quasiquoting constant patterns

Let us start with something less ambitious -
a quasiquoter for constant patterns, allowing us to write

``` { .haskell }
testExpr :: Expr
testExpr = [expr|1+2*3|]

f1 :: Expr -> String
f1 [expr| 1 + 2*3 |] = "Bingo!"
f1 _ = "Sorry, no bonus"

main = putStrLn $ f1 testExpr
```

This can be done similarly to the quasiquoter for expressions:

* record the current position in Haskell file (for parse error reporting);
* parse the expression into our abstract syntax;
* convert our abstract syntax to its Template Haskell representation.

# Building pattern AST

This time we need to construct Template Haskell pattern representation:

``` haskell
quoteExprPat :: String -> TH.Q TH.Pat
quoteExprPat s = do
  pos <- getPosition
  exp <- parseExpr pos s
  dataToPatQ (const Nothing) exp

```

The functions `quoteExprExp` and `quoteExprPat` differ in two respects:

* use `dataToPatQ` instead of `dataToExpQ`
* the result type is different (obviously)

# Antiquotation

The quasiquotation mechanism we have seen so far allows us to translate domain-specific code into Haskell and "inject" it into our program.

Antiquotation, as the name suggests goes in the opposite direction: embeds Haskell entities (e.g. variables) in our DSL.

This sounds complicated, but isn't really. Think HTML templates:

``` { .html}
<html>
<head>
<title>#{pageTitle}
<body><h1>#{pageTitle}
```

The meaning is hopefully obvious --- the value of program variable `pageTitle` should be embedded in the indicated places. In our expression language we might want to write

```
twice :: Expr -> Expr
twice e = [expr| $e + $e |]

testTwice = twice [expr| 3 * 3|]
```

This is nothing revolutionary. Haskell however, uses variables not only in expressions, but also in patterns, and here the story becomes a little interesting.

# Extending quasiquoters

Recall the pattern quasiquoter:

``` { .haskell }
quoteExprPat :: String -> Q Pat
quoteExprPat s = do
  pos <- getPosition
  exp <- parseExpr pos s
  dataToPatQ (const Nothing) exp
```

The `(const Nothing)` is a placeholder for extensions to the standard `Data` to `Pat` translation:

``` haskell
quoteExprPat :: String -> Q Pat
quoteExprPat s = do
  pos <- getPosition
  exp <- Expr.parseExpr pos s
  dataToPatQ (const Nothing `extQ` antiExprPat) exp
```

What are the "extensions"?

# What’s a function extension?
You have

* a generic function, say
```
gen :: Data a => a -> R
```
* a type-specific function, say
```
spec :: T -> R
```

You want a generic function which behaves like spec on values of type T,
and like gen on all other values.

The function `extQ` does just that.

```
extQ :: (Typeable a, Typeable b) => (a -> r) -> (t -> r) -> a -> r

gen `extQ` spec :: Data a => a -> R  -- Data is a subclass of Typeable
```

(NB `extQ` comes from `Data.Generics` and the `Q` in the name has nothing to do with
Template Haskell `Q` monad)

# Extending `dataToPatQ`


```
const Nothing :: b -> Maybe (Q Pat)

extQ :: (Data a, Data t) => (a -> r) -> (t -> r) -> a -> r
-- specialized to Data

antiExprPat :: Expr -> Maybe (Q Pat)

const Nothing `extQ` antiExprPat :: forall b.Data b => b -> Maybe (Q Pat)

dataToPatQ
  :: Data a =>
     (forall b.Data b => b -> Maybe (Q Pat)) -> a -> Q Pat
-- specialized To Expr
-- :: (forall b.Data b => b -> Maybe (Q Pat)) -> Expr -> Q Pat

dataToPatQ (const Nothing `extQ` antiExprPat) :: Expr -> Q Pat
```

# Metavariables
Let us extend our expression syntax and parser with metavariables (variables from the metalanguage):

```haskell
data Expr =  ... | EMetaVar String
           deriving(Show,Typeable,Data)

pExpr :: Parser Expr
pExpr = pTerm `chainl1` spaced addop

pTerm = spaced pFactor `chainl1` spaced mulop
pFactor = pNum <|> pMetaVar

pMetaVar = char '$' >> EMetaVar <$> ident

test1 = parse pExpr "test1" "1 - 2 - 3 * 4 "
test2 = parse pExpr "test2" "$x - $y*$z"
```

# Antiquoting metavariables

The antiquoter is defined as an extension for the `dataToPatQ`:

``` haskell
antiExprPat :: Expr -> Maybe (Q Pat)
antiExprPat (EMetaVar v) = Just $ varP (mkName v)
antiExprPat _ = Nothing
```

* metavariables are translated to `Just` TH variables
* for all the other cases we say `Nothing` - allowing `dataToPatQ` use its default rules

And that's it! Now we can write

``` haskell
eval [expr| $a + $b|] = eval a + eval b
eval [expr| $a * $b|] = eval a * eval b
eval (EInt n) = n
```

# Exercises

* Write a function such that build_ps n generates all projections for n-tuples,
* Write a function `tupleFromList` such that
```
$(tupleFromList 8) [1..8] == (1,2,3,4,5,6,7,8)
```
* Extend the expression simplifier with more rules.
* Add antiquotation to `quoteExprExp`

* Extend the expression quasiquoter to handle metavariables for
  numeric constants, allowing to implement simplification rules like

```
simpl [expr|$int:n$ + $int:m$|] = [expr| $int:m+n$ |]
```
(you are welcome to invent your own syntax in place of `$int: ... $`)

* write a `matrix` quasiquoter such that

```
*MatrixSplice> :{
*MatrixSplice| [matrix|
*MatrixSplice| 1 2
*MatrixSplice| 3 4
*MatrixSplice| |]
*MatrixSplice| :}
[[1,2],[3,4]]
```

be careful with blank lines!

