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

try
    run(`git clone https://github.com/Roboneet/model-zoo.git`)

    run(`ls`)

    cd("/content/model-zoo/")
    run(`git checkout dp`)
    cd("./games/differentiable-programming/trebuchet")


    Pkg.activate(".")
    Pkg.instantiate()
    println("$(@__DIR__), $(pwd())")
    include("$(pwd())/DDPG.jl")

    run(`cp -r ./values/ /content/drive/My\ Drive/trebuchet/values`)
catch e
    println("error occured\n")
    show(e)
    println()
end

cd("/content")
rm("model-zoo", recursive=true, force=true)
