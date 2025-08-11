import matplotlib.pyplot as plt;

"""
Display and close a plot
"""
def __show_plot():
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.show()
    plt.close()

"""
Pre-plot settings that remain constant for all plots
"""
def __preamble():
    plt.figure(figsize=(10, 6))

"""
Plot observed spectrum only, fig1
"""
def observed_spectrum(df, final_plot=True):
    if final_plot == True:
        __preamble();
    
    plt.plot(df['wave_obs_AA'], df['flux_lambda'], label='Observed SED', color='blue')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Brightness (flux per unit wavelength)')

    # Figure 1 specific draws 
    if final_plot == False:
        return;

    plt.title('Observed SED from sample galaxy')
    
    # Highlight low wavelength areas of interest
    emission_start = 4120;
    emission_end = 4200;

    absorption_end = 4245;

    noise_start = 4355;
    noise_end = 4365;

    plt.axvspan(emission_start, emission_end, color='green', alpha=0.3, label='Narrow emission')
    plt.axvspan(emission_end, absorption_end, color='blue', alpha=0.3, label='Absorption')
    # plt.axvspan(noise_start, noise_end, color='red', alpha=0.3, label='Noise')

    __show_plot();
    
"""
Plot observed spectra and model, fig2
"""
def model(df, final_plot = True):
    if final_plot == True:
        __preamble();
    
    # Plot observed spectrum first
    observed_spectrum(df, final_plot=False);

    # Add model line
    model = df["flux_lambda"] - df["residual"];
    plt.plot(df['wave_obs_AA'], model, label='Model SED', color='orange')
    
    if final_plot == False:
        return;
    
    plt.title('Estimated and observed SEDs')

    __show_plot();

"""
Plot raw model residuals, figure 3
"""
def raw_residuals(df, final_plot = True):
    if final_plot == True:
        __preamble();

    plt.plot(df['wave_obs_AA'], df['residual'], label='Residuals (observation minus model)', alpha = 0.3, color='orange')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Residuals (flux per unit wavelength)')

    if final_plot == False:
        return;
    
    plt.title('Model residuals')
    
    # Add moving average of residuals
    window_size = 50;
    min_periods = round(window_size / 2)
    moving_average = df['residual'].rolling(window = window_size, min_periods = min_periods, center = True).mean();
    plt.plot(df['wave_obs_AA'], moving_average, label='Moving average of residuals', color='red')
    
    # Show areas of high moving average at low frequency
    low_frequency_1_start = 3800;
    low_frequency_1_end = 4430;
    plt.axvspan(low_frequency_1_start, low_frequency_1_end, color='blue', alpha=0.3, label='Low frequency wobble')
    
    low_frequency_2_start = 6650;
    low_frequency_2_end = 7325;
    plt.axvspan(low_frequency_2_start, low_frequency_2_end, color='blue', alpha=0.3);
    
    __show_plot();


