{-# LANGUAGE  QuasiQuotes #-}
import ExprQuote2
import Expr2

simpl :: Expr -> Expr
simpl [expr|0 + $x|] = x

main = print $ simpl [expr|0+2|]
