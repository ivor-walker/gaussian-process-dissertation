import matplotlib.pyplot as plt;

import numpy as np;

"""
Display and close a plot
"""
def __show_plot(fig = None):
    if fig is None:
        plt.grid()
        plt.legend()
        plt.tight_layout()

    else:
        fig.tight_layout()

    plt.show()
    return;
    # fig.close()

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

"""
SVGP and celerite comparison
"""
def svgp_vs_celerite(models_data, data_X, data_y, final_plot = True):
    # Horizontal plot of all models
    fig, ax = plt.subplots(1, len(models_data), sharex=True, sharey=True, figsize=(12, 6));
    
    # Add convolution moving average of residuals
    window_size = 50;
    min_periods = round(window_size / 2)
    
    data_X = data_X.ravel();
    data_y = data_y.ravel();
    kernel = np.ones(window_size) / window_size;
    moving_average = np.convolve(data_y, kernel, mode='same');

    # Plot each model in each subplot
    for i, model_data in enumerate(models_data):
        model_name = model_data['model_name'];
        model_y = model_data['result'][0].ravel();
        model_var = model_data['result'][1].ravel();
        
        ax[i].plot(data_X, data_y, label='Original residuals', color='orange', alpha=0.3);
        ax[i].plot(data_X, moving_average, label='Moving average of original residuals', color='red');        
        
        ax[i].plot(data_X, model_y, label="GP residuals", color = 'blue');
        # ax[i].plot(data_X, model_data['mean'], label='GP mean', color='green');
        ax[i].fill_between(data_X, model_y - model_var, model_y + model_var, color='purple', alpha=0.3, label='GP variance');

        ax[i].set_title(model_name);
        ax[i].set_xlabel('Wavelength (Angstrom)');
        ax[i].set_ylabel('Brightness (flux per unit wavelength)');
        ax[i].grid();
        ax[i].legend();
    
    __show_plot(fig);


