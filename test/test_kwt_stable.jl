using Test
using JSON3
include(joinpath(@__DIR__,"..","current_sweep_thermal_nofastscan.jl"))
include(joinpath(@__DIR__,"..","src","gpu_full_curl.jl"))
include(joinpath(@__DIR__,"..","src","gpu_float32_experiment.jl"))
include(joinpath(@__DIR__,"..","src","gpu_kwt_stable_experiment.jl"))

function oracle384(z::ComplexF32,r::ComplexF32,g::Float32,u::Float32)
    setprecision(BigFloat,384) do
        a=BigFloat(real(z)); b=BigFloat(imag(z)); rx=BigFloat(real(r)); ry=BigFloat(imag(r)); gg=BigFloat(g); uu=BigFloat(u)
        d=1+gg^2*(a^2+b^2); m=[1+gg^2*a^2 gg^2*a*b; gg^2*a*b 1+gg^2*b^2]
        x=(m\[rx,ry]) .* (sqrt(d)/uu); (x[1],x[2],d,rx,ry,a,b,gg,uu)
    end
end
cbig(x,y)=complex(Float64(x),Float64(y))

@testset "KWT scalar conditioning oracle" begin
    rows=Any[]; maxpn=0.0; maxpr=0.0; maxheat=0.0; maxres=0.0; maxpnp=0.0; maxprp=0.0; maxhp=0.0; maxrp=0.0; cases=Tuple{ComplexF32,ComplexF32,Float32,Float32}[]
    for amp in Float32[.01,.1,.8,1,3,4], g in Float32[0,4,40,80], ph in Float32[0,pi/4,pi/3,pi/2], B in Float32[0,1e-8,1/(1+g*g*amp*amp),1e-3,1,-1]
        z=ComplexF32(amp*cos(ph),amp*sin(ph)); r=ComplexF32(.37,0)*z+ComplexF32(B,0)*im*z; push!(cases,(z,r,g,Float32(5.79)))
    end
    append!(cases,[(ComplexF32(0),ComplexF32(1,2),Float32(0),Float32(5.79)),(ComplexF32(0),ComplexF32(1,2),Float32(80),Float32(5.79))])
    for (z,r,g,u) in cases
        ox,oy,D,rx,ry,a,b,gg,uu=oracle384(z,r,g,u); om=sqrt(ox^2+oy^2)
        for (name,v) in (("old32",kwt_old32(z,r,g,u)),("stable32",kwt_stable32(z,r,g,u)),("promoted64",kwt_promoted64(z,r,g,u)))
            ex=BigFloat(real(v))-ox; ey=BigFloat(imag(v))-oy; en=sqrt(ex^2+ey^2); norm=Float64(en/(1+om)); rel=om==0 ? Float64(en) : Float64(en/om); rr=sqrt(a^2+b^2); radial=rr==0 ? en : abs((ex*a+ey*b)/rr); tang=rr==0 ? en : abs((-ex*b+ey*a)/rr)
            vx=BigFloat(real(v));vy=BigFloat(imag(v));rad=a*vx+b*vy
            dheat=2*(rx*vx+ry*vy);ideal=2*(rx*ox+ry*oy)
            fullheat=2*uu/sqrt(D)*(vx^2+vy^2+gg^2*rad^2)
            heaterr=Float64(abs(fullheat-ideal)/(1+abs(ideal)))
            dualerr=Float64(abs(dheat-ideal)/(1+abs(ideal)))
            resid=sqrt((uu/sqrt(D)*(vx+gg^2*a*rad)-rx)^2+(uu/sqrt(D)*(vy+gg^2*b*rad)-ry)^2)
            push!(rows,(mode=name,amp=Float64(abs(z)),gamma=Float64(g),normalized_vector_error=norm,relative_vector_error=rel,radial_error=Float64(radial),tangential_error=Float64(tang),heat_normalized_error=heaterr,dual_heat_normalized_error=dualerr,equation_residual=Float64(resid)))
            if name=="promoted64" && om > big"1e-30"; @test rel < 3e-7; elseif name=="promoted64"; @test en < big"3e-7"; end
            maxpn=max(maxpn,norm); maxpr=max(maxpr,rel); maxheat=max(maxheat,heaterr); maxres=max(maxres,Float64(resid))
            if name=="promoted64"; maxpnp=max(maxpnp,norm); maxprp=max(maxprp,rel); maxhp=max(maxhp,heaterr); maxrp=max(maxrp,Float64(resid)); end
        end
        lhs=2*(rx*ox+ry*oy); da=2*(a*ox+b*oy); rhs=2*uu/sqrt(D)*((ox^2+oy^2)+gg^2/4*da^2)
        @test abs(lhs-rhs) / max(abs(lhs),big"1e-100") < big"1e-70"
    end
    qpath=get(ENV,"KWT_CONDITIONING_JSON",""); receipt=(schema="kwt-conditioning-v2",description="384-bit real 2x2 oracle; Float32 values are exact stored inputs; promoted arithmetic is not a whole-trajectory claim",cases=rows,case_count=length(cases),max_all_normalized_error=maxpn,max_all_relative_error=maxpr,max_all_heat_normalized_error=maxheat,max_all_equation_residual=maxres,max_promoted_normalized_error=maxpnp,max_promoted_relative_error=maxprp,max_promoted_heat_normalized_error=maxhp,max_promoted_equation_residual=maxrp,quantization_effect="separate stored-F32 quantization is represented by the F32 input construction; no F64 pre-quantized reference is used")
    @info "KWT conditioning" cases=length(cases) max_promoted_normalized=maxpnp max_promoted_relative=maxprp max_promoted_heat=maxhp max_promoted_residual=maxrp
    isempty(qpath) || open(qpath,"w") do io; write(io,JSON3.write(receipt)); end
end

