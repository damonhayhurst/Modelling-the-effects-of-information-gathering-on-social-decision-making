import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from analyse.get_response_stats import get_is_lie, get_trials_by_condition
from utils.columns import CLUSTER, GAIN_OF_TEN, GAIN_OF_THIRTY, GAIN_UNDER_TEN, GAIN_UNDER_THIRTY, LIE, LOSS_OF_TEN, LOSS_OF_THIRTY, LOSS_UNDER_TEN, LOSS_UNDER_THIRTY, OTHER_LOSS, PID, POSITIVE_GAIN, SELF_GAIN
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
    plot(resid(glmm_model))
    dev.off()
    ''')


def plot_qqline(glmm_model, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.r(f'''
    png("{plotFolder}/qqline.png")
    qqnorm(resid(glmm_model))
    qqline(resid(glmm_model))
    dev.off()
    ''')


def plot_effects(glmm_model, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.r(f'''
    library(effects)
    model_effects <- allEffects(glmm_model)
    png("{plotFolder}/effects.png")
    plot(model_effects, multiline=TRUE)
    dev.off()
    ''')


def plot_interaction(glmm_model, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.r(f'''
    png("{plotFolder}/interaction.png")
    plot(Effect(c("SELFGAIN", "CLUSTER"), glmm_model), multiline = TRUE, main = "Interaction Effect: SELFGAIN * CLUSTER on Probability of Lying")
    dev.off()
    ''')


def plot_profile(glmm_model, plotFolder):
    ro.globalenv['glmm_model'] = glmm_model
    ro.r(f'''
    png("{plotFolder}/profile.png")
    model_eff <- Effect(c('CLUSTER', 'SELFGAIN'), glmm_model)
    plot(model_eff, main="Profile Plot of Predictor1 by Predictor2", xlab="Predictor1", ylab="Response", type="response")
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
        xlab="Fitted Values", 
        ylab="Pearson Residuals", 
        main="Pearson Residuals vs Fitted Values")
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


def prepare_dataframe(analysis_df: DataFrame, cluster_df: DataFrame) -> DataFrame:
    analysis_df[CLUSTER] = cluster_df[CLUSTER].astype('category')
    analysis_df[LIE] = get_is_lie(analysis_df).astype('int')
    analysis_df["SELFGAIN"] = analysis_df[SELF_GAIN]
    analysis_df["OTHERLOSS"] = analysis_df.get(OTHER_LOSS, None)  # Handle cases where OTHER_LOSS might not be present
    analysis_df = analysis_df.reset_index()
    display(analysis_df[[CLUSTER, PID, SELF_GAIN, LIE]])
    return analysis_df


def split_train_test(analysis_df: DataFrame):
    train_df, test_df = train_test_split(analysis_df, test_size=0.1, random_state=42, stratify=analysis_df['LIE'])
    train_df_r = pandas2ri.py2rpy(train_df)
    test_df_r = pandas2ri.py2rpy(test_df)
    return train_df_r, test_df_r


def make_predictions(glmm_model, test_df_r):
    ro.r('''
    predictions <- predict(glmm_model, newdata = test_df_r, type = "response")
    predicted_classes <- ifelse(predictions > 0.5, 1, 0)
    actual_classes <- test_df_r$LIE  # Replace 'Outcome' with your actual column name
    accuracy <- mean(predicted_classes == actual_classes)
    confusion_matrix <- table(predicted_classes, actual_classes)
    print(accuracy)
    print(confusion_matrix)
    ''')

def activate_r_packages():
    pandas2ri.activate()
    lme4 = importr('lme4')
    stats = importr('stats', on_conflict='warn')
    influence_me = importr('influence.ME', on_conflict='warn')
    car = importr('car', on_conflict='warn')
    bp_test = importr('lmtest', on_conflict='warn')
    return lme4, stats, influence_me, car, bp_test


def do_heirarchical_logistical_regression(analysis_df: DataFrame, cluster_df: DataFrame, plotFolder: str, test: bool = False):
    analysis_df = prepare_dataframe(analysis_df, cluster_df)
    lme4, stats, influence_me, car, bp_test = activate_r_packages()

    # Convert the DataFrame to an R dataframe
    df_r = pandas2ri.py2rpy(analysis_df)
    globalenv['df_r'] = df_r

    if test:
        train_df_r, test_df_r = split_train_test(analysis_df)
        globalenv['df_r'] = train_df_r
        globalenv['test_df_r'] = test_df_r

    ro.r('df_r$CLUSTER <- as.factor(df_r$CLUSTER)')

    # Fit the GLMM model using the training data
    ro.r("glmm_model <- glmer(LIE ~ (SELFGAIN + OTHERLOSS) * CLUSTER + (1|PID), data = df_r, family='binomial')")
    glmm_model = ro.globalenv['glmm_model']
    print(ro.r('summary(glmm_model)'))

    if test:
        make_predictions(glmm_model, ro.globalenv['test_df_r'])

    # Plot the GLMM model output and save to file
    plot_glmm_model(glmm_model, plotFolder)

    print(ro.r('vif(glmm_model, type="predictor")'))

    # Calculate influence measures and plot
    plot_influence_measures(glmm_model, plotFolder)

    # Residual plots
    plot_residuals(glmm_model, plotFolder)
    plot_qqline(glmm_model, plotFolder)
    plot_effects(glmm_model, plotFolder)
    plot_interaction(glmm_model, plotFolder)
    plot_profile(glmm_model, plotFolder)
    plot_acf_pacf(glmm_model, plotFolder)
    plot_pearson_residuals_vs_fitted(glmm_model, plotFolder)
    plot_residuals_vs_selgain(glmm_model, df_r, plotFolder)


def do_transformed_heirarchical_logistical_regression(analysis_df: DataFrame, cluster_df: DataFrame, plotFolder: str, test: bool = False):
    analysis_df = prepare_dataframe(analysis_df, cluster_df)
    lme4, stats, influence_me, car, bp_test = activate_r_packages()

    # Convert the DataFrame to an R dataframe
    df_r = pandas2ri.py2rpy(analysis_df)
    globalenv['df_r'] = df_r

    if test:
        train_df_r, test_df_r = split_train_test(analysis_df)
        globalenv['df_r'] = train_df_r
        globalenv['test_df_r'] = test_df_r

    ro.r('df_r$CLUSTER <- as.factor(df_r$CLUSTER)')

    # Fit the GLMM model using the training data
    ro.r("glmm_model <- glmer(LIE ~ I(SELFGAIN^3) * CLUSTER + (1|PID), data = df_r, family='binomial')")
    glmm_model = ro.globalenv['glmm_model']
    print(ro.r('summary(glmm_model)'))

    if test:
        make_predictions(glmm_model, ro.globalenv['test_df_r'])

    # Plot the GLMM model output and save to file
    plot_glmm_model(glmm_model, plotFolder)

    print(ro.r('vif(glmm_model, type="predictor")'))

    # Calculate influence measures and plot
    plot_influence_measures(glmm_model, plotFolder)

    # Residual plots
    plot_residuals(glmm_model, plotFolder)
    plot_qqline(glmm_model, plotFolder)
    plot_effects(glmm_model, plotFolder)
    plot_interaction(glmm_model, plotFolder)
    plot_profile(glmm_model, plotFolder)
    plot_acf_pacf(glmm_model, plotFolder)
    plot_pearson_residuals_vs_fitted(glmm_model, plotFolder)
    plot_residuals_vs_selgain(glmm_model, df_r, plotFolder)


def compute_weights(analysis_df: DataFrame):
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


def do_weighted_heirarchical_logistical_regression(analysis_df: DataFrame, cluster_df: DataFrame, plotFolder: str, test: bool = False, weights: str = None):
    analysis_df = prepare_dataframe(analysis_df, cluster_df)
    
    lme4, stats, influence_me, car, bp_test = activate_r_packages()


    if test:
        train_df_r, test_df_r = split_train_test(analysis_df)
        globalenv['df_r'] = compute_weights(train_df_r)
        globalenv['test_df_r'] = test_df_r
    else:
        analysis_df = compute_weights(analysis_df)
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

    print(ro.r('vif(glmm_model, type="predictor")'))

    # Calculate influence measures and plot
    plot_influence_measures(glmm_model, plotFolder)

    # Residual plots
    plot_residuals(glmm_model, plotFolder)
    plot_qqline(glmm_model, plotFolder)
    plot_effects(glmm_model, plotFolder)
    plot_interaction(glmm_model, plotFolder)
    plot_profile(glmm_model, plotFolder)
    plot_acf_pacf(glmm_model, plotFolder)
    plot_pearson_residuals_vs_fitted(glmm_model, plotFolder)
    plot_residuals_vs_selgain(glmm_model, df_r, plotFolder)


def do_categorical_heirarchical_logistical_regression(analysis_df: DataFrame, cluster_df: DataFrame, plotFolder: str, test: bool = False):
    analysis_df = prepare_dataframe(analysis_df, cluster_df)
    lme4, stats, influence_me, car, bp_test = activate_r_packages()

    # Convert the DataFrame to an R dataframe
    df_r = pandas2ri.py2rpy(analysis_df)
    globalenv['df_r'] = df_r

    if test:
        train_df_r, test_df_r = split_train_test(analysis_df)
        globalenv['df_r'] = train_df_r
        globalenv['test_df_r'] = test_df_r

    ro.r('df_r$CLUSTER <- as.factor(df_r$CLUSTER)')

    # Fit the GLMM model using the training data
    ro.r("glmm_model <- glmer(LIE ~ (SELFGAIN + OTHERLOSS) * CLUSTER + (1|PID),data = df_r,family = 'binomial')")
    glmm_model = ro.globalenv['glmm_model']
    print(ro.r('summary(glmm_model)'))

    if test:
        make_predictions(glmm_model, ro.globalenv['test_df_r'])

    # Plot the GLMM model output and save to file
    plot_glmm_model(glmm_model, plotFolder)

    print(ro.r('vif(glmm_model, type="predictor")'))

    # Calculate influence measures and plot
    plot_influence_measures(glmm_model, plotFolder)

    # Residual plots
    plot_residuals(glmm_model, plotFolder)
    plot_qqline(glmm_model, plotFolder)
    plot_effects(glmm_model, plotFolder)
    plot_interaction(glmm_model, plotFolder)
    plot_profile(glmm_model, plotFolder)
    plot_acf_pacf(glmm_model, plotFolder)
    plot_pearson_residuals_vs_fitted(glmm_model, plotFolder)
    plot_residuals_vs_selgain(glmm_model, df_r, plotFolder)


def do_gam_analysis(analysis_df: DataFrame, cluster_df: DataFrame, test: bool = False):
    plotFolder = "R_gam"
    os.makedirs("plots/plotFolder", exist_ok=True)  # Ensure the plot folder exists

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
    plot_interaction(gam_model, plotFolder)
    plot_profile(gam_model, plotFolder)
    plot_acf_pacf(gam_model, plotFolder)
    plot_pearson_residuals_vs_fitted(gam_model, plotFolder)
    plot_residuals_vs_selgain(gam_model, df_r, plotFolder)
