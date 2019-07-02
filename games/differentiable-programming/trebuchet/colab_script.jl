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

run(`git clone https://github.com/Roboneet/model-zoo.git`)

run(`ls`)

cd("/content/model-zoo/")
run(`git checkout dp`)
cd("./games/differentiable-programming/trebuchet")


Pkg.activate(".")
Pkg.instantiate()
println("$(@__DIR__)")
include("$(@__DIR__)/DDPG.jl")

run(`cp -r ./values/ /content/drive/My\ Drive/trebuchet/values`)
