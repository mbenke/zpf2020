module Conquer where

data Foo = Foo
data Bar = Bar Foo

conquer :: [Foo] -> [Bar]
conquer fs = concatMap (pure . step) fs

step :: Foo -> Bar
step = _
