# include DRSOM.jl from the most up-to-date dev branch.
import Pkg

try
    Pkg.rm("DRSOM")
catch
end

Pkg.develop(url="https://github.com/bzhangcw/DRSOM.jl.git#for-alm")