{- OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}
module RNN where

import TypedFlow
import TypedFlow.Python

atShape :: forall s t. T s t -> T s t
atShape x = x

mycnn :: Gen (T '[120] Int32 -> T '[] Int32 -> ModelOutput Float32 '[2] '[])
mycnn = do
  filters1 <- parameterDefault "f1"
  filters2 <- parameterDefault "f2" 
  w <- parameterDefault "w"
  embs <- parameterDefault "embs"
  return $ \input gold ->
    let --nn :: T '[112] Int32 -> T '[2] Float32
        nn = dense @2 w                      .
             flattenAll                      .
             maxPool1D @40                 .
--             atShape @'[57,40] .
             relu . conv @40 @'[7] filters2  .
--             atShape  @'[5,40].
             maxPool1D @3 .      
--             atShape  @'[114,40].             
             relu . conv @40 @'[7] filters1  .
             mapT (embedding @100 @13314 embs) 
        logits = nn input
    in sparseCategoricalDensePredictions logits gold

main :: IO ()
main = do
  generateFile "model.py" (compile @100 defaultOptions mycnn)
  putStrLn "done!"
