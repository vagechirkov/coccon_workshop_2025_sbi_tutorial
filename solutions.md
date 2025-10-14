```python
# @title Solution for Exercise 1
# Define the observations for each athlete
observation_A = torch.tensor([50.3, 12.1]) # Power thrower
observation_B = torch.tensor([12.1, 9.5])  # High-arc specialist

print(f"Athlete A -> Distance: {observation_A[0]}m, Height: {observation_A[1]}m")
print(f"Athlete B -> Distance: {observation_B[0]}m, Height: {observation_B[1]}m")

# Sample from the posterior for each observation
print("\n📈 Drawing posterior samples for each athlete...")
posterior_samples_A = posterior.sample((10000,), x=observation_A)
posterior_samples_B = posterior.sample((10000,), x=observation_B)
print(f"✅ Drew {len(posterior_samples_A)} samples for each observation.")

# Visualize the two posterior distributions side-by-side
# We use the true parameters for the "Strong Throw" and "High-Arc Throw" as reference points
fig, axes = pairplot(
    [posterior_samples_A, posterior_samples_B],
    points=[strong_throw_params.unsqueeze(0), high_arc_params.unsqueeze(0)],
    upper="contour",
    upper_kwargs=dict(levels=[0.5, 0.8, 0.99]),
    labels=param_names_3d,
    figsize=(8, 8),
);

# Add a legend to distinguish the athletes
plt.legend(handles=axes[1,1].get_legend_handles_labels()[0],
           labels=["Athlete A (Power)", "Athlete B (High-Arc)"],
           frameon=False, fontsize=12, loc="upper left")
plt.suptitle("Comparing Posteriors for Two Athletes", fontsize=16, y=1.02)
plt.show()

print("\n--- Analysis for Athlete A (Power Thrower) ---")
_ = analyze_posterior_statistics(posterior_samples_A, param_names_3d, strong_throw_params)
print("\n--- Analysis for Athlete B (High-Arc Specialist) ---")
_ = analyze_posterior_statistics(posterior_samples_B, param_names_3d, high_arc_params)
```


```python
# @title Solution for Exercise 2
# --- Step 1: Define a new 4D Prior and Parameters ---
prior_4d = create_ball_throw_prior(include_wind=True)
param_names_4d = ["v₀ (velocity)", "θ (angle)", "μ (friction)", "W (wind)"]

# Let's define the true parameters that generated our new observation
# A strong throw, but with a significant tailwind
true_params_with_wind = torch.tensor([25.0, 0.7, 0.1, 3.0]) # v0, angle, friction, wind

# Generate the observation using these "true" parameters
observation_with_wind = ball_throw_simulator(true_params_with_wind)
print(f"🎯 Observed data (with wind): Distance={observation_with_wind[0]:.1f}m, Height={observation_with_wind[1]:.1f}m")

# --- Step 2: Retrain the SBI Model in 4D ---
print("\n⚙️ Retraining SBI pipeline for 4 parameters...")
simulator_4d = process_simulator(ball_throw_simulator, prior_4d, False)
npe_4d = NPE(prior=prior_4d)

num_simulations_4d = 4000 # More parameters often requires more simulations
theta_4d, x_4d = simulate_for_sbi(
    simulator_4d,
    prior_4d,
    num_simulations=num_simulations_4d,
    num_workers=num_workers,
)

print(f"✅ Generated {len(theta_4d)} simulations.")
print("🧠 Training new neural posterior estimator...")
npe_4d.append_simulations(theta_4d, x_4d).train()
print("✅ Training complete!")

# --- Step 3: Build Posterior and Sample ---
posterior_4d = npe_4d.build_posterior()
posterior_samples_4d = posterior_4d.sample((10000,), x=observation_with_wind)
print("✅ Drew posterior samples for the 4D case.")

# --- Step 4: Analyze and Visualize ---
fig, axes = pairplot(
    [posterior_samples_4d],
    points=true_params_with_wind.unsqueeze(0),
    labels=param_names_4d,
    figsize=(10, 10),
);
plt.suptitle("Posterior with Unknown Wind Parameter", fontsize=16, y=1.02)
plt.show()

_ = analyze_posterior_statistics(posterior_samples_4d, param_names_4d, true_params_with_wind)
```

```python
# @title Solution for Exercise 4
use_autocorrelation = False

# --- Step 2: Calculate Summary Statistics from Real Data ---
# We will use the more informative autocorrelation stats
real_observed_summary = summarize_simulation(real_data_timeseries, use_autocorrelation=use_autocorrelation)
real_data_labels = get_summary_labels(use_autocorrelation=use_autocorrelation)
print("\n📋 Summary Statistics from Real Data:")
for label, value in zip(real_data_labels, real_observed_summary):
    print(f"{label:20s}: {value:.2f}")

# @title Solution for Exercise 4 - Inference and Prediction
# --- Step 3: Define a new Prior and SBI pipeline for the Real Data ---
# The dynamics might be different, so let's use a slightly wider prior
lower_bound_real = torch.tensor([0.01, 0.001, 0.001, 0.01])
upper_bound_real = torch.tensor([0.5, 0.5, 0.5, 0.5])
prior_real = BoxUniform(low=lower_bound_real, high=upper_bound_real)

# The simulator needs to match the timeframe of the real data (33 years)
# And use the first data point as the initial condition y0
initial_pop_real = real_data_timeseries[0]
def simulator_real_data(params, use_autocorrelation=use_autocorrelation): 
    populations = lotka_volterra_simulation(
        params,
        t_span=33, # Match the 33 years of data
        dt=1,      # One step per year
        y0=initial_pop_real
    )
    return summarize_simulation(populations, use_autocorrelation)

simulator_real_data = process_simulator(simulator_real_data, prior_real, False)

# --- Step 4: Run the Inference ---
print("⚙️ Running SBI for real-world data. This is computationally intensive...")
npe_real = NPE(prior=prior_real)
num_simulations_real = 10000 # Real data is noisy; more simulations help.

theta_real, x_real = simulate_for_sbi(simulator_real_data, prior_real,
                                      num_simulations=num_simulations_real,
                                      num_workers=num_workers)

print("🧠 Training neural posterior estimator...")
npe_real.append_simulations(theta_real, x_real).train()
posterior_real = npe_real.build_posterior()

print("\n📈 Sampling from posterior conditioned on real data...")
observed_data_real = torch.tensor(real_observed_summary, dtype=torch.float32)
posterior_samples_real = posterior_real.sample((10000,), x=observed_data_real)

# --- Step 5: Analyze the Results ---
print("\n🎉 Inference on real data complete!")
fig = pairplot(posterior_samples_real, labels=[r"$\alpha$", r"$\beta$", r"$\delta$", r"$\gamma$"], figsize=(9, 9))
plt.suptitle("Posterior for Saxony Wildlife Data", fontsize=16, y=1.02)
plt.show()

_ = analyze_posterior_statistics(posterior_samples_real, lv_param_names)

# --- Step 6: Predict the Future! ---
print("\nForecasting future population dynamics...")
# We need a simulator that runs for a longer time to see the future
future_time_span = 60 # years
simulator_for_prediction = lambda params: lotka_volterra_simulation(
    params, t_span=future_time_span, dt=1, y0=initial_pop_real
)

map_sim_real, pred_sims_real = generate_posterior_predictive_simulations(
    posterior=posterior_real,
    observed_data=observed_data_real,
    simulate_func=simulator_for_prediction,
    prior=prior_real,
    num_simulations=1000,
)

# Create the time axes for plotting
time_axis_future = np.arange(1991, 1991 + future_time_span)
times = np.arange(1991, 1991 + len(foxes)) # Assuming 'foxes' holds the historical data

# Calculate uncertainty bounds
lower_bound_real = torch.quantile(pred_sims_real, 0.05, dim=0)
upper_bound_real = torch.quantile(pred_sims_real, 0.95, dim=0)

# --- Create Figure with Two Subplots ---
# 2 rows, 1 column, sharing the x-axis, and a larger figure size
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
fig.suptitle("Prediction of Fox and Rabbit Populations in Saxony", fontsize=20, y=0.93)


# --- Plot 1: Rabbit Population (Prey) ---
ax1.set_title("Rabbit Population (Prey)", fontsize=16)

# Plot historical rabbit data
ax1.plot(times, rabbits, 'x--', color='darkred', label="Historical Rabbits")
# Plot MAP rabbit prediction
ax1.plot(time_axis_future, map_sim_real[:, 0], '--', color='red', lw=2, label="MAP Prediction")
# Plot 90% credible interval for rabbits
ax1.fill_between(time_axis_future, lower_bound_real[:, 0], upper_bound_real[:, 0],
                 color='red', alpha=0.2, label="90% Credible Interval")
# Add a vertical line to mark the start of the prediction
ax1.axvline(2023.5, color='black', linestyle=':', lw=2, label="Prediction Start")
ax1.set_ylabel("Population Proxy (in thousands)", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle='--', alpha=0.5)


# --- Plot 2: Fox Population (Predator) ---
ax2.set_title("Fox Population (Predator)", fontsize=16)

# Plot historical fox data
ax2.plot(times, foxes, 'o-', color='darkblue', label="Historical Foxes")
# Plot MAP fox prediction
ax2.plot(time_axis_future, map_sim_real[:, 1], '--', color='blue', lw=2, label="MAP Prediction")
# Plot 90% credible interval for foxes
ax2.fill_between(time_axis_future, lower_bound_real[:, 1], upper_bound_real[:, 1],
                 color='blue', alpha=0.2, label="90% Credible Interval")
# Add the vertical line to the second plot as well
ax2.axvline(2023.5, color='black', linestyle=':', lw=2, label="Prediction Start")
ax2.set_xlabel("Year", fontsize=14)
ax2.set_ylabel("Population Proxy (in thousands)", fontsize=12)
ax2.legend(loc="upper left")
ax2.grid(True, linestyle='--', alpha=0.5)

# --- Display the Plots ---
plt.tight_layout(rect=[0, 0, 1, 0.92]) # Adjust layout to make room for suptitle
plt.show()

```