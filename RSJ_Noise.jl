using DifferentialEquations
using Distributions
using DelimitedFiles
using ProgressMeter

function phase_lock_areas!(omega, D, A_step)

	@show(omega, D, A_step)

	function RSJ!(du, u, p, t)
		omega, A, B, D = p
		du[1] = -1/2*(conj(u[2])-u[2]*u[1]^2)+1im*(B+A*cos(omega*t))*u[1]
		du[2] = -1/2*(1-abs(u[2])^2)*conj(u[1])
		nothing
	end

	function noise!(du, u, p, t)
		omega, A, B, D = p
		du[1] = sqrt(2*D)
		du[2] = 0
		nothing
	end
	
	function lyapunov!(omega, A, B, D)
		t0 = 2*pi/omega
		zeta_w = [1+1im*0.0, 0+1im*0.0]
		tspan = (0., t0)
		
		p = (omega, A, B, D)
		prob = SDEProblem(RSJ!, noise!, zeta_w, tspan, p)
		sol = solve(prob)
		zeta = last(sol.u)[1]
		w = last(sol.u)[2]
		
		numerator = (1+zeta+sqrt((1-zeta)^2+4*zeta*abs(w)^2))
		denominator = (1+zeta-sqrt((1-zeta)^2+4*zeta*abs(w)^2))
		lyap = abs(numerator/denominator)
		
		return lyap
	end

	A_min = 0.0;
	A_max = 4.0;

	A_values = range(A_min, A_max, step=A_step) |> collect;

	B_min = 0.0;
	B_max = 4.0;

	B_values = range(B_min, B_max, step=A_step) |> collect;

	A_size = length(A_values);
	B_size = length(B_values);

	layp_values = zeros(B_size, A_size);

	@showprogress Threads.@threads for i=1:B_size  
		for j=1:A_size
			layp_values[i,j] = lyapunov!(omega, A_values[j], B_values[i], D)
		end
	end

	name_pattern = join(["ph_lock_D_", string(D), "_omega_", string(omega), "_step_", string(A_step)])
	name_pattern = replace(name_pattern, "." => "_")
	name = join([name_pattern, ".csv"])

	writedlm(name,  layp_values, ',')
	
end

omega = parse(Float64, ARGS[1])
D = parse(Float64, ARGS[2])
A_step = parse(Float64, ARGS[3])
phase_lock_areas!(omega, D, A_step)

