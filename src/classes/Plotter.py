import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcol

class Plotter:
    def __init__(self, model):
        """
        Initialise the Plotter with an A1Model object.
        """
        self.model = model
        self.params = model.params


    def figure_2(self):
        """
        Recreate Figure 2 from the paper.
        """
        p = self.params
        fig, ax = plt.subplots(3, p.P, figsize=(15, 7.5), gridspec_kw={'height_ratios': [2, 1, 1]}, sharex=False, sharey=False)

        time_axis = np.arange(0, p.sim_duration, p.dt)
        P_min, P_max = 0, 15

        for Q in range(P_min, P_max):
            # Plot network activity
            ax[0, Q].plot(time_axis, np.mean(self.model.activity.E[:, Q, :], axis=1))
            ax[0, Q].set_title(f"Col. {Q + 1}")
            ax[0, Q].set_ylim([0, 80])
            if Q == P_min:
                ax[0, Q].set_ylabel("Network Activity (Hz)")
            else:
                ax[0, Q].set_yticklabels([])

            # Plot input amplitude
            ax[1, Q].plot(np.arange(0, p.sim_duration, p.stim_step_size),
                          np.mean(self.model.sensory_input_matrix[:, Q, self.model.spont_index:p.N_E], axis=1))
            ax[1, Q].set_ylim([-0.5, 4.5])
            if Q == P_min:
                ax[1, Q].set_ylabel("Input Amplitude (Hz)")
            else:
                ax[1, Q].set_yticklabels([])

        # Compute max amplitude of s for each column
        max_s = np.max(np.mean(self.model.sensory_input_matrix[:, :, self.model.spont_index:p.N_E], axis=2), axis=0)

        # Remove the last row of subplots to make space for ax2
        for Q in range(p.P):
            fig.delaxes(ax[2, Q])

        # Create a new axis that spans all columns
        ax2 = plt.subplot2grid((3, p.P), (2, P_min), colspan=(P_max - P_min))
        ax2.plot(range(P_min, P_max), max_s[P_min:P_max], marker='o', linestyle='-')
        ax2.set_xlabel("Column")
        ax2.set_ylabel("Max s Amplitude")
        ax2.set_ylim([0, 4.5])
        ax2.set_xlim([-0.35, 14.35])
        ax2.set_xticks(range(0, 15))
        ax2.set_xticklabels(range(1, 16))

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.show()


    def figure_3(self, E_act_store, s_act_store):
        """
        Recreate Figure 3 from the paper.
        """
        matplotlib.rcParams["figure.dpi"] = 300
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["axes.formatter.use_mathtext"] = True
        plt.rcParams["font.size"] = 14

        fig, ax = plt.subplots(8, 1, figsize=(6, 6), sharex=False, sharey=False, gridspec_kw={'hspace': 0.5})

        custom_cmap = matplotlib.colormaps.get_cmap("jet")
        vmin, vmax = 2, 78

        labels = [r"$\bf{VII}$", r"$\bf{VI}$", r"$\bf{V}$", r"$\bf{IV}$", r"$\bf{III}$", r"$\bf{II}$", r"$\bf{I}$"]

        for i, (E_act, label) in enumerate(zip(E_act_store, labels)):
            im = ax[i].imshow(E_act, aspect="auto", cmap=custom_cmap, vmin=vmin, vmax=vmax,
                              origin='lower', interpolation='nearest', extent=[0, 15, 0, 50])
            ax[i].set_ylim(0, 50)
            ax[i].set_yticks([0, 50])
            ax[i].set_xticks([])
            ax[i].set_yticklabels([r"$0$", r"$50$"], fontsize=14)
            ax[i].text(14.5, 49, label, va="bottom", ha="center", fontsize=12)

        fig.text(0.04, 0.5, r"$\bf{Time}$ [$\bf{ms}$]", va="center", rotation="vertical", fontsize=16)

        P_min, P_max = 0, 15
        colors = ["b", "k", "r"]

        for s_act, color in zip(s_act_store, colors):
            ax[-1].plot(range(P_min, P_max), s_act[P_min:P_max], linestyle='-', color=color, linewidth=1.5)

        ax[-1].set_ylim(0, 10)
        ax[-1].set_xlabel(r"$\bf{Column} \ \bf{Index}$", fontsize=16)
        ax[-1].set_ylabel(r"$\bf{Input} \ \bf{Amp.}$", fontsize=16)
        ax[-1].set_xlim([-0.35, 14.35])
        ax[-1].set_xticks([3, 7, 11])
        ax[-1].set_xticklabels([r"$4$", r"$8$", r"$12$"], fontsize=14)
        ax[-1].set_yticks([0, 8])
        ax[-1].set_yticklabels([r"$0$", r"$8$"], fontsize=14)

        ax[-1].spines['top'].set_visible(False)
        ax[-1].spines['right'].set_visible(False)

        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), orientation="vertical", shrink=0.355, aspect=7.5,
                            anchor=(0.0, 0.2), pad=0.01)
        cbar.set_ticks([20, 60])
        cbar.set_ticklabels([r"$20 \ \text{Hz}$", r"$60 \ \text{Hz}$"], fontsize=10)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(0.5)

        #plt.savefig("Fig3.pdf", dpi=300, bbox_inches="tight")

        plt.show()


    def figure_4(self, E_weak, I_weak, E_strong, I_strong, output):
        """
        Recreate Figure 4 from the paper.
        """
        # --- Figure 4A: Excitatory and Inhibitory Responses --- #
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        for idx, (E_data, I_data, title) in enumerate([(E_weak, I_weak, "Weak"), (E_strong, I_strong, "Strong")]):
            E_diff = np.diff(np.mean(E_data, axis=1), axis=0)
            I_diff = np.diff(np.mean(I_data, axis=1), axis=0)

            E_thrshld = 0.4 * np.max(E_diff)
            I_thrshld = 0.4 * np.max(I_diff)

            E_onset = (np.where(E_diff >= E_thrshld)[0][0]) * self.model.params.dt * 1000
            I_onset = (np.where(I_diff >= I_thrshld)[0][0]) * self.model.params.dt * 1000

            time_axis = np.arange(0, self.model.params.sim_duration * 1000, self.model.params.dt * 1000)

            ax[idx].plot(time_axis, np.mean(E_data, axis=1), label='Excitatory', color='blue')
            ax[idx].plot(time_axis, np.mean(I_data, axis=1), label='Inhibitory', color='red')
            ax[idx].axvline(10, color='k', linestyle='--')
            ax[idx].axvline(E_onset, color='blue', linestyle='--')
            ax[idx].axvline(I_onset, color='red', linestyle='--')
            ax[idx].set_ylabel('Firing rate (Hz)')
            ax[idx].set_ylim([0, 80])
            ax[idx].set_title(f'{title} Stimulus Response')
            ax[idx].legend()

        ax[-1].set_xlabel('Time (ms)')
        plt.tight_layout()

        # --- Figure 4B: Latency of PS vs. Stimulus Amplitude --- #
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(output[:, 0], output[:, 1], marker='o', color='blue', label="Excitatory")
        ax.scatter(output[:, 0], output[:, 2], marker='o', color='red', label="Inhibitory")

        ax.set_xlabel("Tone Amplitude (Hz)")
        ax.set_ylabel("Latency to Response (ms)")
        ax.set_ylim([0, 40])
        ax.legend()

        plt.tight_layout()
        plt.show()

    def figure_6(self, outputs, time_axes, ratio_A, ratio_F, ISI, A_vals, c_vals):
        """
        Recreate Figure 6 from the paper.
        """
        fig, ax = plt.subplots(3, 1, figsize=(5, 10), gridspec_kw={'hspace': 0.3})

        # Panel A: Network Activity
        for output, time_axis in zip(outputs, time_axes):
            ax[0].plot(time_axis, np.mean(output[:, 7, :], axis=1), color='black', linewidth=0.8)
        ax[0].set_ylabel('Network Activity [Hz]', fontsize=12, fontweight='bold')
        ax[0].set_xlim([-0.1, 4.2])
        ax[0].set_ylim([-0.1, 85])
        ax[0].tick_params(axis='both', labelsize=10)

        # Panel B: P2/P1 Ratio vs. Input Amplitude
        markers = ['+', 'o', 's']  # Different markers for each amplitude
        linestyles = ['dashed', 'dotted', 'solid']  # Line styles to differentiate
        for i in range(ratio_A.shape[0]):
            ax[1].plot(ISI, ratio_A[i], marker=markers[i], linestyle=linestyles[i], color='black',
                       label=f'Input Amplitude {A_vals[i]}', markersize=6)
        ax[1].set_ylabel('P2/P1 Ratio', fontsize=12, fontweight='bold')
        ax[1].set_xlim([-0.1, 4.2])
        ax[1].set_ylim([0, 1])
        ax[1].tick_params(axis='both', labelsize=10)
        ax[1].legend(fontsize=10, loc='lower right', frameon=False)

        # Panel C: P2/P1 Ratio vs. Column Number
        markers = ['+', 'o', 's', 'x', 'd']  # Different markers for each column
        linestyles = ['dashed', 'dotted', 'solid', 'dashdot', (0, (3, 1, 1, 1))]  # Matching line styles
        for i in range(ratio_F.shape[0]):
            ax[2].plot(ISI, ratio_F[i], marker=markers[i], linestyle=linestyles[i], color='black',
                       label=f'Column {c_vals[i] + 1}', markersize=6)
        ax[2].set_xlabel(r'Inter-Stimulus Interval [$\tau_{\mathrm{rec}}$]', fontsize=12, fontweight='bold')
        ax[2].set_ylabel('P2/P1 Ratio', fontsize=12, fontweight='bold')
        ax[2].set_xlim([-0.1, 4.2])
        ax[2].set_ylim([0, 1])
        ax[2].tick_params(axis='both', labelsize=10)
        ax[2].legend(fontsize=10, loc='lower right', frameon=False)

        plt.show()

    def figure_10(self, act):
        plt.figure(figsize=(6, 6))
        custom_cmap = matplotlib.colormaps.get_cmap("jet")

        vmin, vmax = 2, 78
        im2 = plt.imshow(act, aspect="auto", cmap=custom_cmap, vmin=vmin, vmax=vmax, origin='lower',
                         interpolation='nearest', extent=[0, self.model.params.sim_duration, 0.5, 15.5])
        cbar = plt.colorbar(im2, orientation="vertical", shrink=0.8, pad=0.02)
        cbar.set_ticks([20, 60])
        cbar.set_ticklabels(["20 Hz", "60 Hz"])
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(0.5)

        plt.show()


    def figure_13(self):
        """
        Recreate Figure 13 from the paper.
        """
        matplotlib.rcParams["figure.dpi"] = 300
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["axes.formatter.use_mathtext"] = True
        plt.rcParams["font.size"] = 14

        # Compute the mean activity
        E_act_mean = np.mean(self.model.activity.E, axis=2)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False, gridspec_kw={'wspace': 0.1})
        custom_cmap = matplotlib.colormaps.get_cmap("jet")

        # Left plot (stimulus matrix)
        vmin, vmax = 0, 8
        im1 = ax[0].imshow(self.model.stim_matrix, aspect='auto', cmap=custom_cmap,
                           interpolation='nearest', vmin=vmin, vmax=vmax, origin='lower',
                           extent=[0, 2.5, 0.5, 15.5])
        ax[0].set_xlabel(r"$\bf{Time}$ [$\tau_{\mathrm{rec}}$]", fontsize=20)
        ax[0].set_ylabel(r"$\bf{Column \ Index}$", fontsize=20)
        ax[0].set_xticks([0.5, 1, 1.5, 2])

        ax[0].set_xticklabels([r"$0.5$", r"$1$", r"$1.5$", r"$2$"], fontsize=12)
        ax[0].set_yticks([2, 8, 14])
        ax[0].set_yticklabels([r"$2$", r"$8$", r"$14$"], fontsize=12)

        cbar1 = fig.colorbar(im1, ax=ax[0], orientation="vertical", pad=0.02)

        cbar1.set_ticks([1, 4, 7])
        cbar1.set_ticklabels([r"$1 \ \text{Hz}$", r"$4 \ \text{Hz}$", r"$7 \ \text{Hz}$"], fontsize=12)

        cbar1.ax.yaxis.set_minor_locator(MultipleLocator(1))

        cbar1.ax.tick_params(which='both', direction='in', length=2)
        cbar1.outline.set_edgecolor('black')
        cbar1.outline.set_linewidth(0.5)

        # Right plot (E_act_mean)
        vmin, vmax = 2, 78
        im2 = ax[1].imshow(E_act_mean.T, aspect="auto", cmap=custom_cmap,
                           vmin=vmin, vmax=vmax, origin='lower',
                           interpolation='nearest', extent=[0, 2.5, 0.5, 15.5])
        ax[1].set_xticks([0.5, 1, 1.5, 2])

        ax[1].set_xticklabels([r"$0.5$", r"$1$", r"$1.5$", r"$2$"], fontsize=12)
        ax[1].set_yticks([])

        cbar2 = fig.colorbar(im2, ax=ax[1], orientation="vertical", pad=0.02)
        cbar2.set_ticks([10, 40, 70])
        cbar2.set_ticklabels([r"$10 \ \text{Hz}$", r"$40 \ \text{Hz}$", r"$70 \ \text{Hz}$"], fontsize=12)

        cbar2.ax.yaxis.set_minor_locator(MultipleLocator(10))

        cbar2.ax.tick_params(which='both', direction='in', length=2)
        cbar2.outline.set_edgecolor('black')
        cbar2.outline.set_linewidth(0.5)

        plt.show()

    def figure_11(self, activity, time_steps):
        time_axes = np.arange(time_steps) * (self.model.params.dt / self.model.params.tau_rec)

        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["axes.formatter.use_mathtext"] = True

        plt.figure(figsize=(6, 2.2))
        plt.plot(time_axes, activity, color='black', linewidth=1)

        # Proper LaTeX-style label using mathtext
        plt.xlabel(r"$\bf{Time}$ [$\tau_{\mathrm{rec}}$]", fontsize=20)
        plt.ylabel(r"$\bf{Network \ Activity}$ [$\bf{Hz}$]", fontsize=16)

        plt.xticks([1, 2, 3, 4], [r"$1$", r"$2$", r"$3$", r"$4$"], fontsize=16)
        plt.yticks([20, 60], [r"$20$", r"$60$"], fontsize=16)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)

        ax.tick_params(direction='in', length=2, width=1)
        plt.grid(False)
        plt.tight_layout()

        plt.savefig("LS.png", dpi=300, bbox_inches="tight")

        plt.show()

    def ssa(self, figure_data):

        # --- Plot --- #
        p = self.params
        fig, ax = plt.subplots(2, 2, figsize=(44, 7.5), gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.15, "hspace": 0.2}, sharex=False, sharey=False, )

        time_axis = np.arange(0, p.sim_duration, p.dt)

        ax[0, 0].plot(time_axis, figure_data["mean_E"], color="black")
        ax[0, 0].set_ylim([-10, 80])
        ax[0, 0].set_xlim([-0.5, 35.5])
        ax[0, 0].set_ylabel("Network Activity (Hz)")
        ax[0, 0].set_xlabel("Time (sec)")

        for i, t in enumerate(figure_data["time_steps"]):
            if(figure_data["stimuli_type"][0][i] == 1):
                ax[0, 0].text(t, -8, ' ',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor="cornflowerblue", edgecolor='cornflowerblue', boxstyle='square,pad=0.01'))

            if(figure_data["stimuli_type"][0][i] == 0):
                ax[0, 0].text(t, -4, ' ',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor="lightcoral", edgecolor="lightcoral", boxstyle='square,pad=0.01'))

        mean_x = np.array(figure_data["mean_x"])

        im1 = ax[1, 0].imshow(mean_x[:, ::100], vmax=0.7, aspect='auto', cmap=plt.get_cmap("gray_r"))

        cbar = fig.colorbar(im1, ax=ax[1, 0], orientation = 'horizontal', location="bottom", shrink=0.5, pad=0.02,)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(0.5)
        ax[1, 0].set_ylim([4, 14])
        ax[1, 0].set_xlim([-5, 355])

        ax[1, 0].set_ylabel("Column Index")
        ax[1, 0].set_xlabel("Time (sec)")
        ax[1, 0].set_xticks([])


        ax[0, 1].plot(np.mean(figure_data["deviant_responses1"], axis=1), color="lightcoral")
        ax[0, 1].plot(np.mean(figure_data["standard_responses1"], axis=1), color="cornflowerblue")

        ax[0, 1].set_ylabel("Average Response Activity")
        ax[0, 1].legend(["f1 - 90%", "f2 - 10%"])

        ax[1, 1].plot(np.mean(figure_data["deviant_responses2"], axis=1), color="lightcoral")
        ax[1, 1].plot(np.mean(figure_data["standard_responses2"], axis=1), color="cornflowerblue")

        ax[1, 1].set_xlabel("Time (ms)")
        ax[1, 1].legend(["f2 - 90%", "f1 - 10%"])

        ax[1, 1].text(25, 1.5, '                                                                   ',
        ha='center', va='bottom', fontsize=3, fontweight='bold',
        bbox=dict(facecolor="black", edgecolor='black', boxstyle='square,pad=0.01'))

        ax[1, 1].text(25, 3, 'Tone',
        ha='center', va='bottom', fontsize=8, fontweight='bold',
        bbox=dict(facecolor="white", edgecolor='white', boxstyle='square,pad=0.01'))

        ax[0, 0].spines[['right', 'top']].set_visible(False)
        ax[0, 1].spines[['right', 'top']].set_visible(False)
        ax[1, 0].spines[['right', 'top']].set_visible(False)
        ax[1, 1].spines[['right', 'top']].set_visible(False)

        plt.subplots_adjust(wspace=0.3)
        plt.show()

    def snr(self, act):
        snr_levels = [r"$\bf{+30}$", r"$\bf{+24}$", r"$\bf{+18}$", r"$\bf{+12}$", r"$\bf{+6}$"]

        # --- Create Bar Plot --- #
        fig, ax = plt.subplots(figsize=(7, 3))
        width = 0.07
        x = np.linspace(0, len(snr_levels) - 1, len(snr_levels)) * 0.45
        levels = act.shape[1]

        cmap = cm.get_cmap("coolwarm")
        fixed_colors = [cmap(i) for i in [0.05, 0.28, 0.51, 0.74, 0.97]]
        colors = fixed_colors[:levels]

        for j in range(levels):
            ax.bar(
                x + (j - (levels - 1) / 2) * width, act[:, j], width,
                color=colors[j], edgecolor="black", alpha=1
            )

        for i in range(len(snr_levels)):
            x_positions = [x[i] + (j - (levels - 1) / 2) * width for j in [0, levels - 1]]
            extra = 82 - int((act[i, 0] + act[i, levels - 1]) / 2)
            y_positions = np.array([act[i, 0] + extra, act[i, levels - 1] + extra])

            ax.plot(x_positions, y_positions, color="black", linestyle="-", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(snr_levels, fontsize=14)
        ax.set_ylabel(r"$\bf{Network \ Activity}$ [$\bf{Hz}$]", fontsize=16)
        ax.set_xlabel(r"$\bf{SNR}$ ($\bf{Hz}$)", fontsize=16)
        ax.set_ylim(40, 90)

        plt.yticks([0, 40, 80], [r"$\bf{0}$", r"$\bf{40}$", r"$8\bf{0}$"], fontsize=14)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)

        cmap_fixed = mcol.LinearSegmentedColormap.from_list("custom_cmap", fixed_colors)
        sm = cm.ScalarMappable(cmap=cmap_fixed)
        sm.set_array([])

        cbar_ax = fig.add_axes([0.9, 0.2, 0.015, 0.65])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")

        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([r"$E_{\min}$", r"$E_{\max}$"])
        cbar.ax.tick_params(labelsize=14, length=0)

        ax.tick_params(direction='in', length=5, width=1)
        plt.subplots_adjust(right=0.88, bottom=0.2)
        plt.show()

    def fra(self, act, ratios):
        fig, ax = plt.subplots(1, len(ratios), figsize=(8, 2), sharex=False, sharey=True, gridspec_kw={'wspace': 0.1})
        custom_cmap = matplotlib.colormaps.get_cmap("magma")

        vmin, vmax = 30, 80
        im2 = None
        for i in range(len(ratios)):
            im2 = ax[i].imshow(act[i], aspect="auto", cmap=custom_cmap, vmin=vmin, vmax=vmax, origin='lower',
                               interpolation='nearest', extent=[0.5, 15.5, 0, 4])
            ax[i].set_xticks([])
            if i != 0:
                ax[i].set_yticklabels([])
                ax[i].set_yticks([])
                ax[i].set_ylabel("")
        ax[0].set_yticks([])

        cbar_ax = fig.add_axes([0.92, 0.28, 0.015, 0.6])
        cbar = fig.colorbar(im2, cax=cbar_ax, orientation="vertical")

        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(0.5)
        cbar.set_ticks([40, 70])
        cbar.set_ticklabels([r"$40 \ \text{Hz}$", r"$70 \ \text{Hz}$"], fontsize=10)
        cbar.ax.tick_params(labelsize=14, length=0)

        fig.supxlabel(r"$\bf{Columns}$", fontsize=14, y=0.08)
        fig.supylabel(r"$\bf{Sound} \ \bf{Level}$", fontsize=14, x=0.04, y=0.55)
        fig.subplots_adjust(bottom=0.3)

        plt.show()