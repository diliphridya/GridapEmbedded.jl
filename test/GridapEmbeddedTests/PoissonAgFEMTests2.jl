module PoissonAgFEMTests2

# Solving the Poisson problem on a geometry with boundaries aligned to the boundaries of the background mesh.

using Gridap
using Gridap.Arrays
using GridapEmbedded
using GridapEmbedded.Interfaces
using Test

# Manufactured solution
u(x) = x[1] - x[2]
f(x) = -Δ(u)(x) # Source term
ud(x) = u(x)    # Boundary condition

# Background mesh 
pmin = Point(0.0,0.0)
pmax = Point(1.0,1.0)

n = 7
partition = (n,n)
bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
dp = pmax - pmin
h = dp[1]/n

# Domain
# The key idea is to define a domain that is bigger than actual domain so that it cuts the boundaries of the background mesh. 
# For example, if we want to solve the equation in the quadrilateral domain with vertices at (0.25,0),(0.75,0),(0.75,1),(0.25,1), 
# then we define the geometry as the quadrilateral with vertices at (0.25,-0.25),(0.75,-0.25),(0.75,1.25),(0.25,1.25).

geo = quadrilateral(x0=Point(0.25,-0.25),d1=VectorValue(0.5,0),d2=VectorValue(0,1.5))

# Cut geometry
cutgeo = cut(bgmodel,geo)

# Aggregation strategy
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

# Triangulation and quadrature
Ω_bg = Triangulation(bgmodel)
Ω = Triangulation(cutgeo)

# Extracting the boundaries aligned to the backgound mesh
cutgeo_facets = cut_facets(bgmodel,geo)
Γ1 = BoundaryTriangulation(cutgeo_facets)

# Embedded Boundary
Γ2 = EmbeddedBoundary(cutgeo)

Γ = lazy_append(Γ1,Γ2)
n_Γ = get_normal_vector(Γ)

order = 1
degree = 2*order
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

# FE spaces
model = DiscreteModel(cutgeo)

Vstd = FESpace(model,FiniteElements(PhysicalDomain(),model,lagrangian,Float64,order))
V = AgFEMSpace(Vstd,aggregates)
U = TrialFESpace(V)

# Weak form
const γd = 10.0

a(u,v) =
  ∫( ∇(v)⋅∇(u) ) * dΩ +
  ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u ) * dΓ

l(v) =
  ∫( v*f ) * dΩ +
  ∫( (γd/h)*v*ud - (n_Γ⋅∇(v))*ud ) * dΓ

# Solving
op = AffineFEOperator(a,l,U,V)
uh = solve(op)

# Error
e = u - uh

l2(u) = sqrt(sum( ∫( u*u )*dΩ ))
h1(u) = sqrt(sum( ∫( u*u + ∇(u)⋅∇(u) )*dΩ ))

el2 = l2(e)
eh1 = h1(e)

#colors = color_aggregates(aggregates,bgmodel)
#writevtk(Ω_bg,"trian",celldata=["aggregate"=>aggregates,"color"=>colors],cellfields=["uh"=>uh])
#writevtk(Ω,"trian_O",cellfields=["uh"=>uh])
#writevtk(Γ,"trian_G")
#writevtk(Γ1,"trian_G1")
#writevtk(Γ2,"trian_G2")

@test el2 < 1.e-10
@test eh1 < 1.e-9

end # module
