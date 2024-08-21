import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from analyse.get_response_stats import get_is_lie, get_trials_by_condition
from utils.columns import CLUSTER, FIRST_AOI, GAIN_OF_TEN, GAIN_OF_THIRTY, GAIN_UNDER_TEN, GAIN_UNDER_THIRTY, LAST_AOI, LIE, LOSS_OF_TEN, LOSS_OF_THIRTY, LOSS_UNDER_TEN, LOSS_UNDER_THIRTY, OTHER_LOSS, PID, POSITIVE_GAIN, SELF_GAIN
from utils.display import display
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, data
from rpy2.robjects import globalenv

from pandas import DataFrame


import rpy2.robjects as ro
from pandas import DataFrame
from rpy2.robjects import pandas2ri, globalenv
from rpy2.robjects.packages import importr


def plot_glmm_model(glmm_model, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.r(f'''
    png("{plotFolder}/glmm.png")
    plot(glmm_model)
    dev.off()
    ''')


def plot_influence_measures(glmm_model, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.r(f'''
    influence_measures <- influence(glmm_model, 'PID')
    png("{plotFolder}/influence.png")
    plot(influence_measures)
    dev.off()
    ''')


def plot_residuals(glmm_model, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.r(f'''
    png("{plotFolder}/residuals.png")
    plot(resid(glmm_model, main="", xlab="SELFGAIN", ylab="Residuals"))
    dev.off()
    ''')


def plot_qqline(glmm_model, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.r(f'''
    png("{plotFolder}/qqline.png")
    qqnorm(resid(glmm_model), main="")
    qqline(resid(glmm_model))
    dev.off()
    ''')

def plot_effects(glmm_model, plotFolder, multiline=True, line_colors=None):
    ro.globalenv['glmm_model'] = glmm_model
    multiline = "TRUE" if multiline else "FALSE"
    
    if line_colors is None:
        line_colors_string = ''
    else:
        line_colors_r = ro.StrVector(line_colors)
        ro.globalenv['line_colors'] = line_colors_r
        line_colors_string = 'colors = line_colors'
    
    ro.r(f'''
    library(effects)
    model_effects <- allEffects(glmm_model)
    png("{plotFolder}/effects.png")
    print(model_effects)
    plot(model_effects, multiline={multiline}, lwd=3, main="", {line_colors_string})
    dev.off()
    ''')


def plot_acf_pacf(glmm_model, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.r(f'''
    residuals <- residuals(glmm_model)
    acf_res <- acf(residuals, main="ACF of Residuals")
    png("{plotFolder}/acf.png")
    plot(acf_res)
    dev.off()
    
    pacf_res <- pacf(residuals, main="PACF of Residuals")
    png("{plotFolder}/ppacf.png")
    plot(pacf_res)
    dev.off()
    ''')


def plot_pearson_residuals_vs_fitted(glmm_model, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.r(f'''
    # Calculate Pearson residuals
    pearson_residuals <- residuals(glmm_model, type = "pearson")

    # Calculate fitted values
    fitted_values <- fitted(glmm_model)

    # Plot Pearson residuals vs fitted values
    png("{plotFolder}/pearson_residuals_vs_fitted.png")
    plot(fitted_values, pearson_residuals, 
        xlab="Lie Probability", 
        ylab="Pearson Residuals", 
        main="")
    abline(h=0, col="red")  # Add a horizontal line at 0 for reference
    dev.off()
    ''')


def plot_residuals_vs_selgain(glmm_model, df_r, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.globalenv['df_r'] = df_r
    ro.r(f'''
    # Calculate residuals
    residuals <- residuals(glmm_model, type="response")

    # Extract predictor
    selgain_values <- df_r$SELFGAIN

    # Create a basic plot of residuals vs SELFGAIN
    png("{plotFolder}/residuals_vs_selgain.png")
    plot(selgain_values, residuals, xlab="SELFGAIN", ylab="Residuals", main="Residuals vs SELFGAIN")
    abline(h=0, col="red")  # Adds a horizontal line at 0 for reference
    dev.off()
    ''')


def prepare_dataframe(analysis_df: DataFrame, cluster_df: DataFrame, compute_weights: bool = False) -> DataFrame:
    analysis_df[CLUSTER] = cluster_df[CLUSTER].astype('category')
    analysis_df[FIRST_AOI] = analysis_df[FIRST_AOI].astype('category')
    analysis_df[LAST_AOI] = analysis_df[LAST_AOI].astype('category')
    analysis_df[LIE] = get_is_lie(analysis_df).astype('int')
    analysis_df["SELFGAIN"] = analysis_df[SELF_GAIN]
    analysis_df["OTHERLOSS"] = analysis_df[OTHER_LOSS]
    analysis_df["FIRSTAOI"] = analysis_df[FIRST_AOI]
    analysis_df["LASTAOI"] = analysis_df[LAST_AOI]
    analysis_df = analysis_df.reset_index()
    display(analysis_df[[CLUSTER, PID, SELF_GAIN, LIE]])
    if compute_weights:
        analysis_df = compute_self_gain_weights(analysis_df)
    return analysis_df


def split_train_test(analysis_df: DataFrame, compute_weights: bool = False):
    train_df, test_df = train_test_split(analysis_df, test_size=0.1, random_state=42, stratify=analysis_df['LIE'])
    if compute_weights:
        train_df = compute_self_gain_weights(train_df)
    train_df_r = pandas2ri.py2rpy(train_df)
    test_df_r = pandas2ri.py2rpy(test_df)
    return train_df_r, test_df_r


def make_predictions(glmm_model, test_df_r):
    ro.globalenv['glmm_model'] = glmm_model
    globalenv['test_df_r'] = test_df_r
    ro.r('''
    predictions <- predict(glmm_model, newdata = test_df_r, type = "response")
    predicted_classes <- ifelse(predictions > 0.5, 1, 0)
    actual_classes <- test_df_r$LIE  # Replace 'Outcome' with your actual column name
    accuracy <- mean(predicted_classes == actual_classes)
    confusion_matrix <- table(predicted_classes, actual_classes)
    print(accuracy)
    print(confusion_matrix)
    ''')


def compute_self_gain_weights(analysis_df: DataFrame):
    analysis_df['SELFGAIN_bin'] = pd.qcut(analysis_df['SELFGAIN'], 10, duplicates='drop')
    # Fit an initial model to estimate residuals

    X = sm.add_constant(analysis_df[['SELFGAIN']])  # Model matrix
    model = sm.Logit(analysis_df['LIE'], X).fit(disp=0)
    analysis_df['residuals'] = model.resid_response

    # Calculate variance of residuals within each 'SELFGAIN' bin
    variance_by_bin = analysis_df.groupby('SELFGAIN_bin')['residuals'].var()

    # Map the variance from bins back to the original data rows
    analysis_df['bin_variance'] = analysis_df['SELFGAIN_bin'].map(variance_by_bin).astype(float)
    # Compute weights as inverse of variance
    analysis_df['weights'] = 1 / analysis_df['bin_variance']
    display(analysis_df)
    return analysis_df


def print_vif_stats(glmm_model):
    ro.globalenv['glmm_model'] = glmm_model
    # Extract VIF values
    vif_values = ro.r('vif(glmm_model, type="predictor")')
    predictor_names = ro.r('names(vif(glmm_model, type="predictor"))')
    
    print("VIF:\n")
    for predictor, vif in zip(predictor_names, vif_values):
        print(f"{predictor}:\t {vif}")
    print("\n")


def print_log_likelihood(glmm_model):
    ro.globalenv['glmm_model'] = glmm_model
    print(ro.r('logLik(glmm_model)'))


def print_odds_ratios(glmm_model):
    ro.globalenv['glmm_model'] = glmm_model
    # Extract fixed effects and their exponentiated values (odds ratios)
    fixed_effects = ro.r('fixef(glmm_model)')
    odds_ratios = ro.r('exp(fixef(glmm_model))')
    predictor_names = ro.r('names(fixef(glmm_model))')

    print("Odds Ratios (Effect Sizes): \n")
    for predictor, odds_ratio in zip(predictor_names, odds_ratios):
        print(f"{predictor}:\t {odds_ratio}")
    print("\n")


def activate_r_packages():
    pandas2ri.activate()
    lme4 = importr('lme4')
    stats = importr('stats', on_conflict='warn')
    influence_me = importr('influence.ME', on_conflict='warn')
    car = importr('car', on_conflict='warn')
    bp_test = importr('lmtest', on_conflict='warn')
    return lme4, stats, influence_me, car, bp_test


def ensure_plot_folder_exists(plotFolder: str):
    if not os.path.exists(plotFolder):
        os.makedirs(plotFolder)


def do_heirarchical_logistical_regression(terms: str, analysis_df: DataFrame, cluster_df: DataFrame, plotFolder: str, colors: list[str] = None, multiline: bool = True, vif: bool = True, test: bool = False):
    ensure_plot_folder_exists(plotFolder)
    analysis_df = prepare_dataframe(analysis_df, cluster_df)
    lme4, stats, influence_me, car, bp_test = activate_r_packages()

    # Convert the DataFrame to an R dataframe
    df_r = pandas2ri.py2rpy(analysis_df)
    globalenv['df_r'] = df_r

    if test:
        df_r, test_df_r = split_train_test(analysis_df)
        globalenv['df_r'] = df_r
        globalenv['test_df_r'] = test_df_r
    else:
        df_r = pandas2ri.py2rpy(analysis_df)
        globalenv['df_r'] = df_r

    ro.r('df_r$CLUSTER <- as.factor(df_r$CLUSTER)')

    # Fit the GLMM model using the training data
    ro.r(f"glmm_model <- glmer(LIE ~ {terms} + (1|PID), data = df_r, family='binomial')")
    glmm_model = ro.globalenv['glmm_model']
    print(ro.r('summary(glmm_model)'))

    if test:
        make_predictions(glmm_model, ro.globalenv['test_df_r'])

    # Plot the GLMM model output and save to file
    plot_glmm_model(glmm_model, plotFolder)

    if vif:
        print_vif_stats(glmm_model)

    print_odds_ratios(glmm_model)

    # Calculate influence measures and plot
    plot_influence_measures(glmm_model, plotFolder)

    # Residual plots
    plot_residuals(glmm_model, plotFolder)
    plot_qqline(glmm_model, plotFolder)
    plot_effects(glmm_model, plotFolder, multiline, colors)
    plot_acf_pacf(glmm_model, plotFolder)
    plot_pearson_residuals_vs_fitted(glmm_model, plotFolder)
    plot_residuals_vs_selgain(glmm_model, df_r, plotFolder)



def do_weighted_heirarchical_logistical_regression(analysis_df: DataFrame, cluster_df: DataFrame, plotFolder: str, test: bool = False, weights: str = None):
    ensure_plot_folder_exists(plotFolder)
    analysis_df = prepare_dataframe(analysis_df, cluster_df)

    lme4, stats, influence_me, car, bp_test = activate_r_packages()

    if test:
        df_r, test_df_r = split_train_test(analysis_df, compute_weights=True)
        globalenv['df_r'] = df_r
        globalenv['test_df_r'] = test_df_r
    else:
        df_r = pandas2ri.py2rpy(analysis_df)
        globalenv['df_r'] = df_r

    ro.r('df_r$CLUSTER <- as.factor(df_r$CLUSTER)')

    # Fit the GLMM model using the training data
    if weights:
        ro.r(f"glmm_model <- glmer(LIE ~ SELFGAIN * CLUSTER + (1|PID), data = df_r, family = 'binomial', weights = df_r${weights})")
    else:
        ro.r("glmm_model <- glmer(LIE ~ SELFGAIN * CLUSTER + (1|PID), data = df_r, family = 'binomial')")

    glmm_model = ro.globalenv['glmm_model']
    print(ro.r('summary(glmm_model)'))

    if test:
        make_predictions(glmm_model, ro.globalenv['test_df_r'])

    # Plot the GLMM model output and save to file
    plot_glmm_model(glmm_model, plotFolder)

    print_vif_stats(glmm_model)

    # Calculate influence measures and plot
    plot_influence_measures(glmm_model, plotFolder)

    # Residual plots
    plot_residuals(glmm_model, plotFolder)
    plot_qqline(glmm_model, plotFolder)
    plot_effects(glmm_model, plotFolder)
    plot_acf_pacf(glmm_model, plotFolder)
    plot_pearson_residuals_vs_fitted(glmm_model, plotFolder)
    plot_residuals_vs_selgain(glmm_model, df_r, plotFolder)


def do_categorical_heirarchical_logistical_regression(analysis_df: DataFrame, cluster_df: DataFrame, plotFolder: str, test: bool = False):
    ensure_plot_folder_exists(plotFolder)
    analysis_df = prepare_dataframe(analysis_df, cluster_df)
    lme4, stats, influence_me, car, bp_test = activate_r_packages()

    # Convert the DataFrame to an R dataframe

    if test:
        df_r, test_df_r = split_train_test(analysis_df)
        globalenv['df_r'] = df_r
        globalenv['test_df_r'] = test_df_r
    else:
        df_r = pandas2ri.py2rpy(analysis_df)
        globalenv['df_r'] = df_r

    ro.r('df_r$CLUSTER <- as.factor(df_r$CLUSTER)')

    # Fit the GLMM model using the training data
    ro.r("glmm_model <- glmer(LIE ~ (SELFGAIN + OTHERLOSS) * CLUSTER + (1|PID),data = df_r,family = 'binomial')")
    glmm_model = ro.globalenv['glmm_model']
    print(ro.r('summary(glmm_model)'))

    if test:
        make_predictions(glmm_model, ro.globalenv['test_df_r'])

    # Plot the GLMM model output and save to file
    plot_glmm_model(glmm_model, plotFolder)

    print_vif_stats(glmm_model)

    # Calculate influence measures and plot
    plot_influence_measures(glmm_model, plotFolder)

    # Residual plots
    plot_residuals(glmm_model, plotFolder)
    plot_qqline(glmm_model, plotFolder)
    plot_effects(glmm_model, plotFolder)
    plot_acf_pacf(glmm_model, plotFolder)
    plot_pearson_residuals_vs_fitted(glmm_model, plotFolder)
    plot_residuals_vs_selgain(glmm_model, df_r, plotFolder)


def do_gam_analysis(analysis_df: DataFrame, cluster_df: DataFrame, plotFolder: str, test: bool = False):
    ensure_plot_folder_exists(plotFolder)

    # Prepare the data
    analysis_df = prepare_dataframe(analysis_df, cluster_df)
    pandas2ri.activate()

    # Import necessary R packages via rpy2
    mgcv, stats = activate_r_packages()

    # Convert the DataFrame to an R dataframe
    df_r = pandas2ri.py2rpy(analysis_df)
    globalenv['df_r'] = df_r

    if test:
        train_df_r, test_df_r = split_train_test(analysis_df)
        globalenv['df_r'] = train_df_r
        globalenv['test_df_r'] = test_df_r

    ro.r('df_r$CLUSTER <- as.factor(df_r$CLUSTER)')

    # Fit the GAM model using the training data
    ro.r("gam_model <- gam(LIE ~ s(SELFGAIN) + CLUSTER, data = df_r, family = 'binomial')")
    gam_model = ro.globalenv['gam_model']
    print(ro.r('summary(gam_model)'))

    if test:
        make_predictions(gam_model, ro.globalenv['test_df_r'])

    # Residual plots
    plot_residuals(gam_model, plotFolder)
    plot_qqline(gam_model, plotFolder)
    plot_effects(gam_model, plotFolder)
    plot_acf_pacf(gam_model, plotFolder)
    plot_pearson_residuals_vs_fitted(gam_model, plotFolder)
    plot_residuals_vs_selgain(gam_model, df_r, plotFolder)
