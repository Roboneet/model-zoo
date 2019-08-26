using Flux, Trebuchet
using Flux.Tracker: forwarddiff
using Statistics: mean
using Random
using Plots
using BSON

Random.seed!(0)
lerp(x, lo, hi) = x*(hi-lo)+lo

function shoot(wind, angle, weight)
  Trebuchet.shoot((wind, Trebuchet.deg2rad(angle), weight))[2]
end

shoot(ps) = forwarddiff(p -> shoot(p...), ps)

DIST  = (20, 100)	# Maximum target distance
SPEED =   5 # Maximum wind speed

target() = (randn() * SPEED, lerp(rand(), DIST...))

function DiffRL(i)
  println("DiffRL($i)")
  distance(wind, target) =
    shoot(Tracker.collect([wind, aim(wind, target)...]))

  function loss(wind, target)
    try
      (distance(wind, target) - target)^2
    catch e
      # Roots.jl sometimes give convergence errors, ignore them
      param(0)
    end
  end

  meanloss() = mean(sqrt(loss(target()...)) for i = 1:100)
  lossValues(n=10) = [sqrt(loss(target()...)) for i = 1:n]

  opt = ADAM()

  dataset = (target() for i = 1:1000)
  # p = plot([], [])

  function aim(wind, target)
    angle, weight = model([wind, target])
    angle = σ(angle)*90
    weight = weight + 200
    angle, weight
  end

  model = Chain(Dense(2, 16, σ),
                Dense(16, 64, σ),
                Dense(64, 16, σ),
                Dense(16, 2)) |> f64

  θ = params(model)

  losses = []
  # cb = Flux.throttle(() -> @show(meanloss()), 10)
  cb = () -> begin
    l = lossValues()
    # @show l
    push!(losses, Flux.data.(l))
  end

  Flux.@epochs 10 Flux.train!(loss, θ, dataset, opt, cb = cb)
  BSON.@save "values/DiffRL_losses$i.bson" losses
  losses
end

function manyDiffRL(n)
  lossCollection = []
  for i=1:n
    push!(lossCollection, DiffRL(i))
  end
  lossCollection
end

lossCollection = manyDiffRL(10)

BSON.@save "values/DiffRL_lossCollection.bson" lossCollection
