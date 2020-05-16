module GetURL where
import Network.HTTP

getURL :: String -> IO String
getURL url = simpleHTTP (getRequest url) >>= getResponseBody
