module MonvisoNamedVectorizationsExt

using Monviso, NamedVectorizations

Monviso.proj_variable(model::Model, x::NV) =
    NV{VariableRef}(layout(x), @variable(model, [1:length(x)]))

end # end of MonvisoNamedVectorizationsExt
