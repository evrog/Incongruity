{ nixpkgs ? import <nixpkgs> {} }:
let
   # nixpkgs_source = fetchTarball "https://github.com/NixOS/nixpkgs/archive/9d0b6b9dfc92a2704e2111aa836f5bdbf8c9ba42.tar.gz";
   # nixpkgs_source = /local_dir; # for local directory
   nixpkgs_source = nixpkgs.fetchFromGitHub { # for safety of checking the hash
      owner = "jyp";
      repo = "nixpkgs";
      rev = "cudnn7.3-cuda9.0";
      sha256 = "1jvsagry3842slgzxkqmp66mxs5b3clbpm7xijv1vjp2wxxvispf";
    };
   # nixpkgs_source = ~/repo/nixpkgs;
in
with (import nixpkgs_source {}).pkgs;
let hp = haskellPackages.override{
      overrides = self: super: {
        pretty-compact = self.callPackage ./pretty-compact.nix {};
        typedflow = self.callPackage ./typedflow.nix {};};};
    ghc = hp.ghcWithPackages (ps: with ps; ([ typedflow cabal-install QuickCheck ]));
    py = (pkgs.python36.withPackages (ps: [ps.tensorflowWithoutCuda
       	 			     	   ps.tqdm
					   ps.scikitlearn
					   ps.pandas
					   ps.nltk
                                           ]));
in pkgs.stdenv.mkDerivation {
  name = "my-env-0";
  buildInputs = [ ghc py ];
  shellHook = ''
 export LANG=en_US.UTF-8
 export LC_CTYPE=en_US.UTF-8
 eval $(egrep ^export ${ghc}/bin/ghc)
'';
}

