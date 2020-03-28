
macro publish(mod,name)
  quote
    using GridapEmbedded.$mod: $name; export $name
  end
end

@publish Interfaces cut
@publish Interfaces EmbeddedDiscretization
@publish Interfaces EmbeddedBoundary
@publish Interfaces GhostSkeleton
@publish Interfaces IN
@publish Interfaces OUT

@publish LevelSetCutters LevelSetCutter
@publish LevelSetCutters doughnut
@publish LevelSetCutters tube
@publish LevelSetCutters olympic_rings
@publish LevelSetCutters sphere
@publish LevelSetCutters disc