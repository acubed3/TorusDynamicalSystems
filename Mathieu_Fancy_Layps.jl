using Plots
using DifferentialEquations
using Distributions
using ProgressMeter
using DelimitedFiles

function phase_lock_areas!(omega)

	@show(omega)

	function Mathieu_Mobius!(du, u, p, t)
		omega, A, B = p
		du[1] = -1/2*(conj(u[2])-u[2]*u[1]^2)*(B-1+A*cos(omega*t))-1im*(B+1+A*cos(omega*t))*u[1]
		du[2] = -1/2*(1-abs(u[2])^2)*conj(u[1])*(-1+B+A*cos(omega*t))
		nothing
	end
	
	function lyapunov_exp!(omega, A, B)
		u0 = [exp(1im*0), 0.0]
		t0 = 2*pi/omega
		tspan = (0., t0)
		p = (omega, A, B)


		prob = ODEProblem(Mathieu_Mobius!, u0, tspan, p, abstol=1e-10)
		sol = solve(prob)
		
		w = last(sol.u)[2]
		zeta = last(sol.u)[1]
		
		lambda2 = (1+zeta+sqrt((1-zeta)^2+4*zeta*abs(w)^2))
		lambda1 = (1+zeta-sqrt((1-zeta)^2+4*zeta*abs(w)^2))
		
		if abs(lambda1) > 0.0
			layp_exp = log(abs(lambda2/lambda1))
		else
			layp_exp = log(abs(lambda2/(lambda1+1e-8)))
		end 
		return layp_exp
	end

	A_min = -4.0;
	A_max = 0.0;
	A_step = 0.005;

	A_values = range(A_min, A_max, step=A_step) |> collect;

	B_min = -4.0;
	B_max = +4.0;
	B_step = 0.005;

	B_values = range(B_min, B_max, step=B_step) |> collect;

	A_size = length(A_values);
	B_size = length(B_values);

	layp_exps = zeros(B_size, A_size);

	@showprogress Threads.@threads for i=1:B_size  
		for j=1:A_size
			layp_exps[i,j] = lyapunov_exp!(omega, A_values[j], B_values[i])
		end
	end

	name_pattern = join(["Mathieu_FancyLyaps_omega_", string(omega), "_step_", string(A_step)])
	name_pattern = replace(name_pattern, "." => "_")
	name = join([name_pattern, ".csv"])

	writedlm(name,  layp_exps, ',')
	
end

omega = parse(Float64, ARGS[1])
phase_lock_areas!(omega)