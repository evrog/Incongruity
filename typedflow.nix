{ mkDerivation, base, ghc-typelits-knownnat, mtl, pretty-compact, fetchgit
, stdenv
}:
mkDerivation {
  pname = "typedflow";
  version = "0.1";
  src = ./TypedFlow;
  libraryHaskellDepends = [
    base ghc-typelits-knownnat mtl pretty-compact
  ];
  description = "Typed frontend to TensorFlow and higher-order deep learning";
  license = stdenv.lib.licenses.lgpl3;
}
