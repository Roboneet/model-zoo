# after running the install julia script,
# mount drive with trebuchet folder
# upload this script to /content
# and run `include("colab_script.jl")`

using Pkg
pkg"st"

using Flux
using CuArrays

# should work
cu(rand(2, 2)) + cu(rand(2, 2))

;git clone https://github.com/Roboneet/model-zoo.git
;git checkout dp

;ls

;cd model-zoo/games/differentiable-programming/trebuchet

Pkg.activate(".")
Pkg.instantiate()
include("DDPG.jl")

;cp -r ./values/ /content/drive/My\ Drive/trebuchet/values
