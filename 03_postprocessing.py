import numpy as np
import os
import pandas as pd
import numpy as np
import tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import genextreme, kurtosis, skew
from utils.FIP_Calc import FIPCalculations
from matplotlib.widgets import Slider, Button
from matplotlib.offsetbox import AnchoredText
import json
import sys

from decimal import Decimal
import re
import ast

class Sliderlog(Slider):

    """Logarithmic slider.

    Takes in every method and function of the matplotlib's slider.

    Set slider to *val* visually so the slider still is lineat but display 10**val next to the slider.

    Return 10**val to the update function (func)"""

    def set_val(self, val):

        xy = self.poly.xy
        if self.orientation == 'vertical':
            xy[1] = 0, val
            xy[2] = 1, val
        else:
            xy[2] = val, 1
            xy[3] = val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % 10**val)   # Modified to display 10**val instead of val
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.items():
                func(10**val)

if __name__ == "__main__":
    woehler = True
    stromeyer = True
    Smax = True # Flag if stressvalues in list are s_max. SN diagramm shows S_amp
    R = -1                                                                                  #####
    with open ('./Config.json') as json_file:
        cfg = json.load(json_file)
    #### USER INPUT ####
    root = cfg["root"]     
    analysisType = cfg["AnalysisType"][0]
    FIP_df = pd.DataFrame()
    stats_AB_df = pd.DataFrame(columns=['sigma', 'A', 'B', 'd_pmps'])    
    d_gr_list = list()
    min_N_rves = 10000
    j = 0
    pow_alpha_min = -10
    pow_alpha_max = 10
    pow_alpha_0 = -8
    u_min = -10
    u_max = 10
    u_0 = 0
    n_bins = 15
    
    exp_params = {'R0p1':[2e-12, 5.788], 'R0p5':[8e-12,5.7781]}
    if not "CrackProp" in analysisType :
        cfg = cfg["PostProcessing"][analysisType]
        rRatio = cfg["R_Ratio"]
        material = cfg["MaterialName"]
        stepType =  cfg["step_type"]
        Load_Horizont = cfg["Loadings"]
        Load_Horizont_float = re.findall(r'\d+', str(Load_Horizont))
        Load_Horizont_float = [float(h) for h in Load_Horizont_float]
        
        #if load horizont = smax adjust R and uncomment next two lines
        if Smax:
            
            #Load_Horizont_float = [(h - h*R)/2 for h in Load_Horizont_float]
            Load_Horizont_float = [h for h in Load_Horizont_float]
        len_horizont = len(Load_Horizont)
        while j < len_horizont:
           #### 
            horizont = Load_Horizont[j]
            Results_data = f"{root}CSVFiles"
            pythonDir = f"{root}PythonControl/"
            postprocDir = f"{root}Postprocessing/"
            n_cycles_path = f'{root}CSVFiles/ncycles.csv'                                              #####create "ncycles.csv" in the same folder
            #n_cycles = pd.read_csv(n_cycles_path, names=['JobID','n_cycles'])
            n_cycles = 6
            if not os.path.isdir(postprocDir):
                os.makedirs(postprocDir)
            P_List = list()
            for i in range(int(len(os.listdir(Results_data))/3)+1):
                if i < 10:
                    counter = "00"+str(i)
                elif 9 < i and i < 99:
                    counter = "0"+str(i)
                else:
                    counter = str(i)
                P_max, d_gr_i = FIPCalculations().get_vals(Results_data, counter)
                
                if P_max > 0:
                    #number_of_sim_cycles = np.mean(n_cycles.loc[n_cycles['JobID'] == i, 'n_cycles'].values)
                    number_of_sim_cycles = 6
                    print('number_of_sim_cycles: ' + str(number_of_sim_cycles))
                    if number_of_sim_cycles > 1:
                        P_List.append(P_max/number_of_sim_cycles)
                        d_gr_list.append(d_gr_i)
            print(f'{len(P_List)}: RVEs were evaluated in Load Horizont {horizont}')
            fig = plt.figure(figsize=[9.6, 7.2])
            plt.tight_layout(pad=1.2, w_pad=2.0, h_pad=2.0)
            if len(P_List) == 0:
                print(Load_Horizont[j] + 'MPa is dropped from analysis.')
                Load_Horizont.pop(j)
                Load_Horizont_float.pop(j)
                len_horizont -= 1
                continue
            else:
                d_pmps = FIPCalculations.calc_delta_pmps(P_List)
                density, bins, ignored = plt.hist(P_List, bins=n_bins , alpha=0.5, density=True, label='simulation data', edgecolor='black', linewidth=1.2)
                bin_centers = bins[:-1] + np.diff(bins) / 2
                popt, _ = curve_fit(FIPCalculations().extreme_val, bin_centers, density, p0=[np.mean(bin_centers), np.std(bin_centers)])
                freq = FIPCalculations().extreme_val(bin_centers, popt[0], popt[1])
                k = kurtosis(freq)
                s = skew(freq) 
                
                print('#######')
                print(popt)
                print(len(Load_Horizont_float))
                print(j)
                print('#######')
                temp_df = pd.DataFrame([[float(Load_Horizont_float[j]), popt[0], popt[1], d_pmps, k, s]], columns=['sigma', 'A', 'B', 'd_pmps','kurtosis','skewness'])
                stats_AB_df = pd.concat([stats_AB_df, temp_df])
                j += 1
            temp_FIP_df = pd.DataFrame(columns=[f'FIP_{horizont}'], data=P_List)
            if len(FIP_df) == 0:
                FIP_df[f'FIP_{horizont}'] = temp_FIP_df[f'FIP_{horizont}']
            elif len(FIP_df) < len(temp_FIP_df):
                batch = temp_FIP_df.sample(len(FIP_df))
                batch = batch.reset_index(drop=True)
                FIP_df[f'FIP_{horizont}'] = batch[f'FIP_{horizont}']
            elif len(FIP_df) >= len(temp_FIP_df):
                FIP_df = FIP_df.sample(len(temp_FIP_df))
                FIP_df = FIP_df.reset_index(drop=True)
                FIP_df[f'FIP_{horizont}'] = temp_FIP_df[f'FIP_{horizont}']
            
            plt.title(r'$P_{mps,max}$ extreme value distribution at $ \sigma_a = $'+horizont, fontsize=24)   ##plot extreme value distribution
            plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
            plt.tick_params(axis ='both', labelsize=20)
            plt.xlabel(r'$P_{mps,max}$', fontsize=22)
            plt.ylabel('density', fontsize=22)

            x_interval_for_fit = np.linspace(bins[0], bins[-1], 1000)
            plt.plot(x_interval_for_fit, FIPCalculations().extreme_val(x_interval_for_fit, A=popt[0], B=popt[1]), label='extreme value distribution')
            plt.xlim([min(P_List)-(bins[-1]-bins[-2]), max(P_List)+bins[-1]-bins[-2]])
            #plt.xlim([0.09, 0.15])
            plt.legend(fontsize=22)
            plt.savefig(f"{postprocDir}/FIP_histo_{horizont}.png")
            plt.close()

            
        if j == 0:
            sys.exit('There was no accumulated plastic strain in all the RVEs')
        FIP_df.to_csv(f"{postprocDir}/FIPS.csv")                                                        
        stats_AB_df.to_csv(f"{postprocDir}/sig_AB.csv")                                                
        d_gr = np.mean(d_gr_list)
        
        print(stats_AB_df)
        FIP_obj = FIPCalculations(stats_AB_df['sigma'], stats_AB_df['A'], stats_AB_df['B'], stats_AB_df['kurtosis'], stats_AB_df['skewness'], stats_AB_df['d_pmps'], d_gr)
        N_hcf = FIP_obj.n_hcf(alpha=10**pow_alpha_0, u=u_0)
        print("******")
        print(N_hcf)
        print("******")
        print(stats_AB_df['sigma'].to_numpy().flatten())

        # Draw the initial plot
        # The 'line' variable is used for modifying the line later
        axis_color = 'lightgoldenrodyellow'
        SN_Plot, ax = plt.subplots(figsize=(9.6, 7.2))
        bbox_props = dict(boxstyle="round, pad=0.3", fc="white", ec="gray", lw=1)
        [points] = ax.plot(N_hcf, stats_AB_df['sigma'].to_numpy(), 'bo', label='simulation data')
        # plot the results
        if woehler:
            x_fit_w, y_fit_w, popt_w = FIP_obj.woehler_fit(N_hcf, stats_AB_df['sigma'].to_numpy().flatten())
            [regression_w] = ax.plot(x_fit_w, y_fit_w, '--k', label='regression with Wöhler Eq.')
        else:
            regression_w = None
        if stromeyer:
            x_fit_s, y_fit_s, popt_s  = FIP_obj.stromeyer_fit(N_hcf, stats_AB_df['sigma'].to_numpy().flatten())
            [regression_s] = ax.plot(x_fit_s, y_fit_s, '-.g', label='regression with Stromeyer Equation')
        else:
            regression_s = None
        try:
            if stromeyer:
                experimental_df_s = pd.read_csv(f"{postprocDir}/exp_SN_s.csv", header=0)                  
                x_fit_s_exp, y_fit_s_exp, popt_s_exp  = FIP_obj.stromeyer_fit(experimental_df_s['N'].to_numpy().flatten(), experimental_df_s['Stress'].to_numpy().flatten())
                ax.plot(experimental_df_s['N'], experimental_df_s['Stress'], 'ok', label='experimental data')
                ax.plot(x_fit_s_exp, y_fit_s_exp, '-g', label='experimental stromeyer fit')
                text_box_x = 0.9*min(x_fit_s_exp)
                text_box_y = 0.9*min(y_fit_s_exp)
                stddev_s = np.sqrt(np.mean((y_fit_s_exp - y_fit_s)**2))
            else:
                stddev_s = '---'
                y_fit_s_exp = None
            if woehler: 
                experimental_df_w = pd.read_csv(f"{postprocDir}/exp_SN_w.csv", header=0)                   
                x_fit_w_exp, y_fit_w_exp, popt_s_exp  = FIP_obj.woehler_fit(experimental_df_w['N'].to_numpy().flatten(), experimental_df_w['Stress'].to_numpy().flatten())
                #ax.plot(experimental_df_w['N'], experimental_df_w['Stress'], 'xg', label='experimental data')
                ax.plot(x_fit_w_exp, y_fit_w_exp, '-k', label='experimental Wöhler fit')
                stddev_w = np.sqrt(np.mean((y_fit_w_exp - y_fit_w)**2))
                text_box_x = 0.9*min(x_fit_w_exp)
                text_box_y = 0.9*min(y_fit_w_exp)
            else:
                stddev_w = '---'
                y_fit_w_exp = None
            
            
            #text = ax.text(text_box_x, text_box_y, 
            #               f"standard dev_w = {stddev_w}"+"\n" 
            #               + f"standard dev_s = {stddev_s}", 
            #           ha="left", va="top", size=14, bbox=bbox_props)  # ha: horizontal alignment, va: vertical alignment
        except Exception as e:
            #text = ax.text(1e5, 2e2, rf"standard dev. = Ex. data is missing", 
            #           ha="left", va="bottom", size=14, bbox=bbox_props)
            print('Whoops something went wrong with the experimental data: ', e)
        
        anchored_text = AnchoredText(f"standard dev_w = {stddev_w}"+"\n" 
                           + f"standard dev_s = {stddev_s}", loc=2)
        at = ax.add_artist(anchored_text)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.set_xlim([1000, 1e8])
        ax.set_xscale('log')
        ax.set_ylim([100, 1000])
        ax.set_yscale('log')
        
        plt.grid(visible=True, which='minor', color='lightgray', linestyle='-')
        plt.grid(visible=True, which='minor', color='lightgray', linestyle=':')

        # Add two sliders for tweaking the parameters

        # Define an axes area and draw a slider in it
        alpha_slider_ax = SN_Plot.add_axes([0.25, 0.1, 0.45, 0.03], facecolor=axis_color)
        alpha_slider = Slider(alpha_slider_ax, r'$\alpha$', pow_alpha_min, pow_alpha_max, valinit=pow_alpha_0)
        alpha = Decimal(10**pow_alpha_0)
        alpha_slider.valtext.set_text(f"{alpha:.2E}")

        # Draw another slider
        u_slider_ax = SN_Plot.add_axes([0.25, 0.05, 0.45, 0.03], facecolor=axis_color)
        u_slider = Slider(u_slider_ax, 'u', u_min, u_max, valinit=u_0)
        
        # Add a button to save the image
        save_button_ax = SN_Plot.add_axes([0.81, 0.05, 0.1, 0.075])
        save_button = Button(save_button_ax, 'Save')

        FIP_obj = FIPCalculations(sigma=stats_AB_df['sigma'], A=stats_AB_df['A'], B=stats_AB_df['B'], 
                                  k=stats_AB_df['kurtosis'], s=stats_AB_df['skewness'], d_pmps=stats_AB_df['d_pmps'], d_gr=d_gr, fig=SN_Plot, slider_alpha=alpha_slider, 
                                  slider_u=u_slider, points=points, 
                                  regression_s=regression_s, 
                                  regression_w=regression_w, 
                                  text=at, y_fit_s_exp = y_fit_s_exp , y_fit_w_exp = y_fit_w_exp, postprocDir=postprocDir)

        alpha_slider.on_changed(FIP_obj.sliders_on_changed_woehler)
        u_slider.on_changed(FIP_obj.sliders_on_changed_woehler)

        save_button.on_clicked(FIP_obj.save_button_on_clicked)
        ax.set_xlabel(r'Number of cycles', fontsize=16)
        ax.set_ylabel(r'Stress amplitude $\sigma_a (MPa)$', fontsize=16)
        ax.tick_params(axis ='both', labelsize=14)
        ax.legend(fontsize=14)
        SN_Plot.tight_layout(pad=1.2, w_pad=2.0, h_pad=2.0)
        SN_Plot.subplots_adjust(bottom=0.25)
        plt.show()
        
        

    else:
        cfg = cfg["PostProcessing"][analysisType]
        analysisType = "CrackProp"
        rRatio=cfg["R_Ratio"]
        material=cfg["MaterialName"]
        stepType=cfg["step_type"]
        dDload=cfg["dDload"]
        W =cfg["width"]/1e3
        B =cfg["thickness"]/1e3
        A0 =B*W
        crackLengthInit=cfg["crackLengthInit"]/1e3
        crackLength=cfg["CrackLength"]

        crackLength_float = [0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02]
        j = 0
        while j < len(crackLength_float):
            crack_str = crackLength[j]
            Results_data = f"{root}SimulationResults/{material}/{analysisType}/{stepType}/{rRatio}/{crack_str}/00_Data"
            pythonDir = f"{root}PythonControl/"
            postprocDir = f"{root}Postprocessing/{material}/{analysisType}/{stepType}/{rRatio}"
            n_cycles_path = f'{Results_data}/ncycles.csv'                                                        
            n_cycles = pd.read_csv(n_cycles_path, names=['JobID','n_cycles'])
            if not os.path.isdir(postprocDir):
                os.makedirs(postprocDir)
            P_List = list()
            for i in range(int(len(os.listdir(Results_data))/3)+1):
                if i < 10:
                    counter = "00"+str(i)
                elif 9 < i and i < 99:
                    counter = "0"+str(i)
                else:
                    counter = str(i)
                P_max, d_gr_i = FIPCalculations().get_vals(Results_data, counter)
                if P_max > 0:
                    number_of_sim_cycles = np.mean(n_cycles.loc[n_cycles['JobID'] == i, 'n_cycles'].values)
                    print('number_of_sim_cycles: ' + str(number_of_sim_cycles))
                    if number_of_sim_cycles > 1:
                        P_List.append(P_max/number_of_sim_cycles)
                        d_gr_list.append(d_gr_i)
            print(f'{len(P_List)} RVEs were evaluated for crack {crack_str}')
            if len(P_List) == 0:
                crackLength.pop(j)
                crackLength_float.pop(j)
                continue
            else:
                alpha_K = 2*crackLength_float[j]/W
                dK = dDload/B*np.sqrt(np.pi*alpha_K/(2*W)*1/(np.cos(np.pi*alpha_K/2)))*1e-6
                sigma_i = dDload/(A0-(crackLength_float[j]+crackLengthInit)*B)
                d_pmps = FIPCalculations().calc_delta_pmps(P_List)
                density, bins, ignored = plt.hist(P_List, bins=n_bins , alpha=0.5, density=True, label='simulation data', edgecolor='black', linewidth=1.2)
                bin_centers = bins[:-1] + np.diff(bins) / 2
                popt, _ = curve_fit(FIPCalculations().extreme_val, bin_centers, density, p0=[0.00125, 0.000125])
                freq = FIPCalculations().extreme_val(bin_centers, popt[0], popt[1])
                k = kurtosis(freq)
                s = skew(freq) 
                temp_df = pd.DataFrame([[crackLength_float[j], sigma_i, popt[0], popt[1], d_pmps, dK, k, s]], columns=['crack','sigma', 'A', 'B', 'd_pmps','dK', 'kurtosis','skewness'])
                #temp_df = pd.DataFrame([[crackLength_float[j], sigma_i, popt[0], popt[1], d_pmps, dK]], columns=['crack', 'sigma', 'A', 'B', 'd_pmps', 'dK'])
                stats_AB_df = pd.concat([stats_AB_df, temp_df])
                j += 1
            temp_FIP_df = pd.DataFrame(columns=[f'FIP_{crack_str}'], data=P_List)
            if len(FIP_df) == 0:
                FIP_df[f'FIP_{crack_str}'] = temp_FIP_df[f'FIP_{crack_str}']
            elif len(FIP_df) < len(temp_FIP_df):
                batch = temp_FIP_df.sample(len(FIP_df))
                batch = batch.reset_index(drop=True)
                FIP_df[f'FIP_{crack_str}'] = batch[f'FIP_{crack_str}']
            elif len(FIP_df) >= len(temp_FIP_df):
                FIP_df = FIP_df.sample(len(temp_FIP_df))
                FIP_df = FIP_df.reset_index(drop=True)
                FIP_df[f'FIP_{crack_str}'] = temp_FIP_df[f'FIP_{crack_str}']
            
            plt.title(r'$P_{mps,max}$ extreme value distribution at $ crack: $'+crack_str, fontsize=20)
            plt.xlabel(r'$P_{mps,max}$', fontsize=16)
            plt.ylabel('density', fontsize=16)
            
            

            x_interval_for_fit = np.logspace(0, 2, 1000)
            plt.plot(x_interval_for_fit, FIPCalculations().extreme_val(x_interval_for_fit, A=popt[0], B=popt[1]), label='extreme value distribution')
            #plt.xlim([min(d_pmps)-0.05*max(d_pmps), max(d_pmps)+0.05*max(d_pmps)])
            plt.xlim([0.09, 1.5])
            plt.legend()
            plt.savefig(f"{postprocDir}/FIP_histo_{crack_str}.png")
            plt.close()

        if j == 0:
            sys.exit('There was no accumulated plastic strain in all the RVEs')
        
        FIP_df.to_csv(f"{root}Postprocessing/{material}/{analysisType}/{stepType}/{rRatio}/FIPS.csv")         
        stats_AB_df.to_csv(f"{root}Postprocessing/{material}/{analysisType}/{stepType}/{rRatio}/sig_AB.csv")    
        d_gr = np.mean(d_gr_list)
        FIP_obj = FIPCalculations(sigma=stats_AB_df['sigma'], A=stats_AB_df['A'], B=stats_AB_df['B'],k=stats_AB_df['kurtosis'], s=stats_AB_df['skewness'],
                                  d_pmps=stats_AB_df['d_pmps'], d_gr=d_gr, dK=stats_AB_df['dK'], crack=stats_AB_df['crack'])
        N_inc = FIP_obj.n_inc(alpha=10**pow_alpha_0)
        extrapolation = FIP_obj.extrapolation(u=u_0)
        stats_AB_df['dadN'] = FIP_obj.dadn(N_inc, extrapolation)
        
        print(stats_AB_df['dK'].to_numpy().flatten())
        print(stats_AB_df['dadN'].to_numpy().flatten())

        x_fit, y_fit, popt_dadn = FIP_obj.crack_prop_fit(stats_AB_df['dK'].to_numpy().flatten(), stats_AB_df['dadN'].to_numpy().flatten())
        exp_dK = np.logspace(1,2,50)
        exp_dadN = FIPCalculations.exponential_curve(exp_dK, exp_params[rRatio][0],exp_params[rRatio][1])
        
        # Draw the initial plot
        # The 'line' variable is used for modifying the line later

        axis_color = 'lightgoldenrodyellow'

        dadN_Plot = plt.figure()
        plt.rcParams['font.size'] = '24'
        plt.rcParams['figure.figsize'] = [12.8,9.6]
        ax = dadN_Plot.add_subplot(111)
        ax.plot(exp_dK,exp_dadN, '--k', label='experimental data')
        a = '%.2E'%Decimal(str(popt_dadn[0]))
        b = str(round(popt_dadn[1],2))
        bbox_props = dict(boxstyle="round, pad=0.3", fc="white", ec="gray", lw=1)
        text = ax.text(1.5, 0.1, rf"$\frac{{da}}{{dN}} = {a} \cdot \Delta K^{b}$", ha="left", va="bottom", size=15, bbox=bbox_props)
        [points] = ax.plot(stats_AB_df['dK'].to_numpy(), stats_AB_df['dadN'].to_numpy(), 'ro', label='simulation data')
        # plot the results
        [regression_dadn] = ax.plot(x_fit, y_fit, '--r', label='regression with simulation data')
        
        # Add two sliders for tweaking the parameters
        

        # Define an axes area and draw a slider in it
        alpha_slider_ax = dadN_Plot.add_axes([0.25, 0.2, 0.55, 0.03], facecolor=axis_color)
        alpha_slider = Slider(alpha_slider_ax, r'$\alpha$', pow_alpha_min, pow_alpha_max, valinit=pow_alpha_0)
        alpha = Decimal(10**pow_alpha_0)
        alpha_slider.valtext.set_text(f"{alpha:.2E}")

        # Draw another slider
        u_slider_ax = dadN_Plot.add_axes([0.25, 0.15, 0.55, 0.03], facecolor=axis_color)
        u_slider = Slider(u_slider_ax, 'u', u_min, u_max, valinit=u_0)
        
        # Add a button for resetting the parameters
        #reset_button_ax = dadN_Plot.add_axes([0.8, 0.025, 0, 0])
        #reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
        FIP_obj = FIPCalculations(sigma=stats_AB_df['sigma'], A=stats_AB_df['A'], B=stats_AB_df['B'], k=stats_AB_df['kurtosis'], s=stats_AB_df['skewness'],
                                  d_pmps=stats_AB_df['d_pmps'], d_gr=d_gr, dK=stats_AB_df['dK'], 
                                  crack=stats_AB_df['crack'], fig=dadN_Plot, slider_alpha=alpha_slider, 
                                  slider_u=u_slider, points=points, regression_dadn=regression_dadn, text=text)
        alpha_slider.on_changed(FIP_obj.sliders_on_changed_dadn)
        u_slider.on_changed(FIP_obj.sliders_on_changed_dadn)
        #reset_button.on_clicked(FIP_obj.reset_button_on_clicked)
        ax.set_xlabel(r'$ \Delta K \left(MPa \sqrt{m}\right)$')
        ax.set_ylabel(r'$ \frac{da}{dN} \left(\frac{mm}{cyc}\right)$')
        ax.legend()
        ax.set_xlim([1, 500])
        ax.set_xscale('log')
        ax.set_ylim([1e-8, 1])
        ax.set_yscale('log')
        plt.show()
