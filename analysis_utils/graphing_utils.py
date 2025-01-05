import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


class InteractionCIPlotter:
    def __init__(
        self,
        regression_model,
        dataframe,
        output_dir,
        interaction_term,
        explanatory_variable,
        dependent_variable,
        control_variables=None,
    ):
        self.regression_model = regression_model
        self.dataframe = dataframe
        self.output_dir = output_dir
        self.interaction_term = interaction_term
        self.explanatory_variable = explanatory_variable
        self.dependent_variable = dependent_variable
        self.control_variables = control_variables if control_variables else []

    def generate_predictions(self, explanatory_range, interaction_value):
        control_means = {
            control_var: self.dataframe[control_var].mean()
            for control_var in self.control_variables
        }

        prediction_data = {
            self.explanatory_variable: explanatory_range,
            self.interaction_term: interaction_value,
            **control_means,
        }

        prediction_df = pd.DataFrame(prediction_data)

        predictions = self.regression_model.get_prediction(
            prediction_df
        ).summary_frame()
        prediction_df["predicted_mean"] = predictions["mean"]
        prediction_df["lower_bound"] = predictions["mean_ci_lower"]
        prediction_df["upper_bound"] = predictions["mean_ci_upper"]

        return prediction_df

    def plot_predictions(self, predictions_all):
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=predictions_all,
            x=self.explanatory_variable,
            y="predicted_mean",
            hue=self.interaction_term,
            palette="tab10",
            linewidth=2,
        )

        unique_interactions = predictions_all[self.interaction_term].unique()
        for interaction_value in unique_interactions:
            interaction_data = predictions_all[
                predictions_all[self.interaction_term] == interaction_value
            ]
            plt.fill_between(
                interaction_data[self.explanatory_variable],
                interaction_data["lower_bound"],
                interaction_data["upper_bound"],
                alpha=0.2,
                label=None,
            )

        plt.xlabel(self.explanatory_variable.replace("_", " ").title())
        plt.ylabel(f"predicted_{self.dependent_variable}".replace("_", " ").title())
        plt.title(
            f"Predicted {self.dependent_variable.replace('_', ' ').title()} vs {self.explanatory_variable.replace('_', ' ').title()} by {self.interaction_term.replace('_', ' ').title()}"
        )
        if self.control_variables:
            control_vars_str = ", ".join(self.control_variables)
            plt.suptitle(
                f"Predicted at the mean of control variables: {control_vars_str}",
                fontsize=10,
                y=0.90,
            )
            plt.subplots_adjust(top=0.85)
        plt.legend(title=self.interaction_term.replace("_", " ").title())
        plt.tight_layout()
        output_path = (
            self.output_dir
            / f"predicted_{self.dependent_variable}_vs_{self.explanatory_variable}_by_{self.interaction_term}.png"
        )
        plt.savefig(output_path)
        plt.show()
        print(f"Plot saved to {output_path.resolve()}")

    def create_plot(self, smoothness=1000):
        explanatory_range = np.linspace(
            self.dataframe[self.explanatory_variable].min(),
            self.dataframe[self.explanatory_variable].max(),
            smoothness,
        )
        unique_interactions = self.dataframe[self.interaction_term].unique()

        predictions_all = pd.DataFrame()
        for interaction_value in unique_interactions:
            prediction_df = self.generate_predictions(
                explanatory_range, interaction_value
            )
            predictions_all = pd.concat(
                [predictions_all, prediction_df], ignore_index=True
            )

        self.plot_predictions(predictions_all)


class ScatterLowessCIPlotter:
    def __init__(self, title, xlabel, ylabel, x_var, y_var, data):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_var = x_var
        self.y_var = y_var
        self.data = data.copy()

    def plot(self, lowess_frac=0.2, save_path=None):
        data = self.data
        # Calculate Lowess smoother
        lowess = sm.nonparametric.lowess
        lowess_results = lowess(data[self.y_var], data[self.x_var], frac=lowess_frac)

        # Extract Lowess results
        lowess_x = lowess_results[:, 0]
        lowess_y = lowess_results[:, 1]

        # Estimate residuals
        residuals = data[self.y_var] - np.interp(data[self.x_var], lowess_x, lowess_y)
        std_error = np.std(residuals)

        # Calculate confidence intervals
        ci = 1.96 * std_error  # 95% confidence interval
        lowess_ci_lower = lowess_y - ci
        lowess_ci_upper = lowess_y + ci

        # Plot scatter plot with Lowess line and confidence intervals
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=self.x_var, y=self.y_var, data=data, alpha=0.3)
        plt.plot(lowess_x, lowess_y, color="red", label="Lowess")
        plt.fill_between(
            lowess_x,
            lowess_ci_lower,
            lowess_ci_upper,
            color="red",
            alpha=0.2,
            label="Lowess CI (std. norm, 95%)",
        )
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()


class ViolinPlotter:
    def __init__(self, data, days_of_week_column, confounding_variables):
        """
        Initialize the ViolinPlotter class.

        Parameters:
        - data: pd.DataFrame, the dataset to plot.
        - days_of_week_column: str, the column with days of the week categories.
        - confounding_variables: list, a list of variable names to explore.
        """
        self.data = data
        self.days_of_week_column = days_of_week_column
        self.confounding_variables = confounding_variables
        # Ensure days of the week are ordered starting from Monday
        self.data[self.days_of_week_column] = pd.Categorical(
            self.data[self.days_of_week_column],
            categories=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            ordered=True,
        )

    def plot(self, figsize=(15, 10), save_path=None):
        """
        Generate and display a single plot with subplots for each variable using Seaborn and Matplotlib.

        Parameters:
        - figsize: tuple, size of the figure.
        - save_path: str, if provided, saves the figure to this path.
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        for i, variable in enumerate(self.confounding_variables):
            sns.violinplot(
                ax=axes[i],
                data=self.data,
                x=self.days_of_week_column,
                y=variable,
                inner="quartile",
            )
            # axes[i].set_title(
            #     f"Violin Plot of {variable} by Day of the Week", fontsize=12
            # )
            axes[i].set_xlabel("Day of the Week")
            axes[i].set_ylabel(variable.replace("_", " ").title())

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(
            "Violin Plots of Confounding Variables by Day of the Week", fontsize=16
        )
        if save_path:
            plt.savefig(save_path)
        plt.show()
