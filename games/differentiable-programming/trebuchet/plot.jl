using Plots
using BSON: load
using Statistics: mean, std

# convert Vec{Vec{Vec{T}}} vectors to Array{T, 3}
_3d_v2m(vecs, x, y, z) =
  reshape(vcat(map(x -> hcat(vcat(hcat(x...)'...)), vecs)'...), x, y, z)

rewardCollection = load("./values/rewardCollection.bson")[:rewardCollection]
lossCollection = load("./values/DiffRL_lossCollection.bson")[:lossCollection]
lossCollection = Float32.(_3d_v2m(lossCollection,
  length(lossCollection), length(lossCollection[1]), length(lossCollection[1][1])))

function findμσ(c::Array{Float32,3})
  r = mean(c, dims=3)
  reshape(mean(r, dims=1), :), reshape(std(r, dims=1), :)
end

rμ, rσ = findμσ(rewardCollection)
lμ, lσ = findμσ(lossCollection)

p=1:10:length(rμ)
plot(p, lμ[p], ribbon=lσ[p], fillalpha=0.5, xlabel="Training steps",
  ylabel="Loss", label="DP")
plot!(p, rμ[p], ribbon=rσ[p], fillalpha=0.5, label="DDPG")
savefig("trebuchet_DPvsDDPG_10000.png")
