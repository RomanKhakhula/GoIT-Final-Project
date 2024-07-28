import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model_evaluation import model_evaluation # type: ignore
from data_analysis import data_analysis, plot_correlation_matrix, prepare_data_for_prediction # type: ignore

# Вкладка 1: Аналіз даних
def data_analysis_streamlit():
    st.title("Аналіз даних:")

    # Run data analysis
    analysis_results, final_df = data_analysis()

    # Create a list of available metrics
    available_metrics = [key for key, _ in analysis_results]
    
    # Select which metric to display
    selected_metric = st.selectbox("Оберіть метрику для відображення", available_metrics)

    # Display selected metric
    for key, value in analysis_results:
        if key == selected_metric:
            st.write(key)
            if isinstance(value, pd.DataFrame):
                st.write(value)
            elif isinstance(value, plt.Figure):
                st.pyplot(value)
            else:
                st.write(value)

    # Define the actual columns in final_df for correlation matrices
    discrete_columns = ['is_tv_subscriber', 'is_movie_package_subscriber', 'reamining_contract',
                        'service_failure_count_0', 'service_failure_count_1', 'service_failure_count_2',
                        'service_failure_count_3', 'service_failure_count_4', 'download_over_limit_0',
                        'download_over_limit_1', 'download_over_limit_2', 'download_over_limit_3',
                        'download_over_limit_4', 'download_over_limit_5', 'download_over_limit_6',
                        'download_over_limit_7', 'churn']

    continuous_columns = ['subscription_age_norm', 'bill_avg_norm', 'download_avg_norm', 'upload_avg_norm', 'churn']

    st.write("Correlation Matrix for Discrete Features")
    st.pyplot(plot_correlation_matrix(final_df, discrete_columns))

    st.write("Correlation Matrix for Continuous Features")
    st.pyplot(plot_correlation_matrix(final_df, continuous_columns))

    # Display final DataFrame
    st.write("Final DataFrame:")
    st.write(final_df.head())

# Вкладка 2: Оцінка моделей
def model_evaluation_streamlit():
    st.title("Оцінка моделей:")
    global scaler
    global models
    model_history, scaler, models = model_evaluation()

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
    st.image(f"D:\\Repos\\Final_GoIT_Project\\{selected_model}.png", caption=f"{selected_model} accuracy\score")
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
            result_df = pd.DataFrame()
            predictions = {}
            for model_name, model in models.items():
                if model_name == 'Neural Network':
                    pred = predict_nn(prepared_data_scaled, model)
                else:
                    pred = predict_ml(prepared_data_scaled, model)
                predictions[model_name] = pred

            for model_name, pred in predictions.items():
                result_df[f'{model_name}_probability'] = pred

            st.write(result_df)
            st.download_button(
                label="Download Predictions",
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
        subscription_age = st.slider("subscription_age", 0.0, 100.0)
        bill_avg = st.slider("bill_avg", 0.0, 200.0)
        reamining_contract = st.slider("reamining_contract", 0.0, 12.0)
        service_failure_count = st.selectbox("service_failure_count", list(range(5)))
        download_avg = st.slider("download_avg", 0.0, 100.0)
        upload_avg = st.slider("upload_avg", 0.0, 100.0)
        download_over_limit = st.selectbox("download_over_limit", list(range(8)))

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
            result_df = pd.DataFrame()
            predictions = {}
            for model_name, model in models.items():
                if model_name == "Neural Network":
                    pred = predict_nn(prepared_data_scaled, model)
                else:
                    pred = predict_ml(prepared_data_scaled, model)
                predictions[model_name] = pred
            
            for model_name, pred in predictions.items():
                result_df[f'{model_name}_probability'] = pred

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
