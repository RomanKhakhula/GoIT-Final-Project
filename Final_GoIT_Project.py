import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from data_analysis import features_groups, analysis_results, final_df, work_df, correlations, prepare_data_for_prediction
from model_evaluation import model_history, scaler, models

# Вкладка 1: Аналіз даних
def data_analysis_streamlit():
    st.title("Аналіз даних:")

    # Create a group of available metrics
    available_metrics_group = st.selectbox('Оберіть групу метрик для відображення:', [key for key in features_groups.keys()])

    # Create a list of available metrics
    available_metrics = features_groups[available_metrics_group]
    
    # Select which metric to display
    selected_metric = st.selectbox("Оберіть метрику для відображення", available_metrics)

    # # Display selected metric

    st.write(f'{selected_metric} (describe):')
    st.write(analysis_results[selected_metric][0])
    st.write(f'{selected_metric} (frequency / avg prob of churn):')
    st.write(analysis_results[selected_metric][1])
    st.write(f'{selected_metric} (frequency / avg prob of churn) vizualization:')
    st.pyplot(analysis_results[selected_metric][2])

    st.title("Аналіз кореляцій:")

    # Define the actual columns in final_df for correlation matrices
    
    selected_correlation = st.selectbox('Оберіть групу метрик для відображення:', [key for key in correlations.keys()])

    st.write(selected_correlation)
    st.pyplot(correlations[selected_correlation][1])

    # Display final DataFrame
    # st.write("Final DataFrame:")
    # st.write(final_df.head())


# Вкладка 2: Оцінка моделей
def model_evaluation_streamlit():
    st.title("Оцінка моделей:")

    # Create a list of available models
    available_models = list(model_history.keys())
    
    # Select which model to display
    selected_model = st.selectbox("Оберіть модель для відображення", available_models)

    # Display selected model's metrics
    metrics = model_history[selected_model]
    st.write(f"Model: {selected_model}")
    st.write(f"Accuracy: {metrics['Accuracy']:.4f}")
    st.write(f"Precision: {metrics['Precision']:.4f}")
    st.write(f"Recall: {metrics['Recall']:.4f}")
    st.write(f"F1 Score: {metrics['F1 Score']:.4f}")
    st.image(str(Path(f'{selected_model}.png').absolute()), caption=f"{selected_model} accuracy\score")
    st.write("---")

# Вкладка 3: Передбачення
def prediction():
    st.title("Передбачення")

    def predict_ml(data, model):
        return model.predict_proba(data)[:, 1]

    def predict_nn(data, model):
        return model.predict(data).flatten()

    def predict_from_csv(uploaded_file):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            prepared_data = prepare_data_for_prediction(df)
            prepared_data_scaled = scaler.transform(prepared_data)
            result_df = None
            predictions = {}
            for model_name, model in models.items():
                if model_name == 'Neural Network':
                    pred = predict_nn(prepared_data_scaled, model)
                else:
                    pred = predict_ml(prepared_data_scaled, model)
                predictions[model_name] = pred

            for model_name, pred in predictions.items():
                if result_df is None:
                    result_df = pd.DataFrame(pred.T, columns = [f'{model_name}_probability'], index = list(prepared_data.index.values))
                else:
                    result_df = pd.concat([result_df, pd.DataFrame(pred.T, columns = [f'{model_name}_probability'], index = list(prepared_data.index.values))], axis = 1)

            result_df = pd.concat([result_df.map('{:,.2%}'.format), work_df['churn']], axis = 1).rename(columns = {'churn' : 'Fact churn'})
            result_df = result_df.loc[result_df[result_df.columns[0]].notna()]
            st.write(result_df)
            st.download_button(
                label="Завантажити предікти",
                data=result_df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

    st.write("Оберіть метод введення даних для передбачення:")
    input_method = st.radio("Метод введення даних", ["Завантажити CSV файл", "Ввести дані вручну"])

    if input_method == "Завантажити CSV файл":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            st.write("Завантажено файл:")
            st.write(uploaded_file.name)
            if st.button("Передбачити з CSV"):
                predict_from_csv(uploaded_file)

    elif input_method == "Ввести дані вручну":
        user_id = st.number_input('id', min_value=0)
        is_tv_subscriber = st.selectbox("is_tv_subscriber", [0, 1])
        is_movie_package_subscriber = st.selectbox("is_movie_package_subscriber", [0, 1])
        subscription_age = st.slider("subscription_age", work_df['subscription_age'].min(), work_df['subscription_age'].max())
        bill_avg = st.slider("bill_avg", work_df['bill_avg'].min(), work_df['bill_avg'].max())
        reamining_contract = st.slider("reamining_contract", work_df['reamining_contract'].min(), work_df['reamining_contract'].max())
        service_failure_count = st.selectbox("service_failure_count", list(range(work_df['service_failure_count'].max() + 1)))
        download_avg = st.slider("download_avg", work_df['download_avg'].min(), work_df['download_avg'].max())
        upload_avg = st.slider("upload_avg", work_df['upload_avg'].min(), work_df['upload_avg'].max())
        download_over_limit = st.selectbox("download_over_limit", list(range(work_df['download_over_limit'].max() + 1)))

        if st.button("Передбачити"):
            input_data = pd.DataFrame([{
            'id': user_id, 'is_tv_subscriber': is_tv_subscriber, 
            'is_movie_package_subscriber': is_movie_package_subscriber, 
            'subscription_age': subscription_age, 'bill_avg': bill_avg,
            'reamining_contract': reamining_contract, 'service_failure_count': service_failure_count, 
            'download_avg': download_avg, 'upload_avg': upload_avg, 'download_over_limit': download_over_limit
            }])
            prepared_data = prepare_data_for_prediction(input_data)
            prepared_data_scaled = scaler.transform(prepared_data)
            result_df = None
            predictions = {}
            for model_name, model in models.items():
                if model_name == "Neural Network":
                    pred = predict_nn(prepared_data_scaled, model)
                else:
                    pred = predict_ml(prepared_data_scaled, model)
                predictions[model_name] = pred
            
            for model_name, pred in predictions.items():
                if result_df is None:
                    result_df = pd.DataFrame(pred.T, columns = [f'{model_name}_probability'], index = list(prepared_data.index.values))
                else:
                    result_df = pd.concat([result_df, pd.DataFrame(pred.T, columns = [f'{model_name}_probability'], index = list(prepared_data.index.values))], axis = 1)

            result_df = pd.concat([result_df.map('{:,.2%}'.format), work_df['churn']], axis = 1).rename(columns = {'churn' : 'Fact churn'})
            result_df = result_df.loc[result_df[result_df.columns[0]].notna()]
            st.write("Прогнози для введеного запису:")
            st.write(result_df)
            st.download_button(
                label="Download Predictions",
                data=result_df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

# Створення вкладок
tab1, tab2, tab3 = st.tabs(["Аналіз даних", "Оцінка моделей", "Передбачення"])

with tab1:
    data_analysis_streamlit()

with tab2:
    model_evaluation_streamlit()

with tab3:
    prediction()