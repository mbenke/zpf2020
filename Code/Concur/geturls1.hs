import GetURL
import Control.Concurrent

main = do
  m1 <- newEmptyMVar
  m2 <- newEmptyMVar
  forkIO $ do
    r <- getURL "http://evemaps.dotlan.net/map/Domain"
    putMVar m1 r

  forkIO $ do
    r <- getURL "http://evemaps.dotlan.net/map/Lonetrek"
    putMVar m2 r

  r1 <- takeMVar m1
  putStrLn "1 DONE"
  r2 <- takeMVar m2
  putStrLn "2 DONE"
