import io
import streamlit as st
import os
import pandas as pd
from streamlit_extras.dataframe_explorer import dataframe_explorer

st.title("Regression Models")

def display_nearest_neighbors():
    st.subheader("En Yakın Komşular Regresyonu")
    st.write("""
    En Yakın Komşular Regresyonu, özellik alanındaki en yakın veri noktalarını bulma prensibiyle çalışır.
    Bu, regresyon görevleri için kullanılan parametrik olmayan bir yöntemdir.

    - Hedef değişkeni, en yakın komşuların değerlerini ortalayarak tahmin eder.
    - Mesafe metriği seçimine duyarlıdır.
    
    ## Hiperparametreler:
    - **Komşu Sayısı**: Dikkate alınacak en yakın komşuların sayısını kontrol eder.
    - **Mesafe Metriği**: Yaygın seçimler arasında Öklid, Manhattan vb. bulunur.
    """)

    # Kullanıcıların modeli denemesi için kaydırıcılar veya giriş kutuları ekleyebilirsiniz

    # Tahmin için etkileşim veya örnek girişler ekleyin

def display_svr():
    st.subheader("SVR (Destek Vektör Regresyonu)")
    st.write("""
    SVR (Destek Vektör Regresyonu), regresyon gerçekleştirmek için Destek Vektör Makineleri prensiplerini kullanır.
    
    - Hatanın belirli bir eşik içinde kalmasını sağlamaya çalışır ve özellikle yüksek boyutlu veriler için faydalıdır.
    - Giriş verisini daha yüksek boyutlu alanlara haritalamak için çekirdek hilelerini kullanarak doğrusal olmayan verilerle iyi çalışabilir.
    
    ## Ana Parametreler:
    - **C**: Hata ile model karmaşıklığı arasındaki dengeyi kontrol eden düzenleme parametresi.
    - **Çekirdek**: Yaygın çekirdek türleri arasında doğrusal, polinom ve RBF (Radial Basis Function) bulunur.
    """)

    # SVR için `C` ve `epsilon` gibi parametreler için kaydırıcılar ekleyebilirsiniz

def display_random_forest():
    st.subheader("Rastgele Orman Regresyonu")
    st.write("""
    Rastgele Orman Regresyonu, regresyon için birden fazla karar ağacı kullanan bir topluluk öğrenme yöntemidir.
    
    - Birden fazla ağacın tahminlerini ortalayarak varyansı ve aşırı uyumu azaltır.
    - Verideki karmaşık ilişkileri yakalayabilir.
    
    ## Ana Özellikler:
    - **Ağaç Sayısı**: Daha fazla ağaç, modeli iyileştirir, ancak hesaplama süresinin artmasına neden olur.
    - **Maksimum Derinlik**: Her ağacın maksimum derinliği; daha büyük bir derinlik aşırı uyuma yol açabilir.
    """)

def display_linear_regression():
    st.title("Linear Regression")
    st.info("Yetkili olduğunuz veri setleri:")

    #train_file = st.text_input("Train set", placeholder=".csv, .txt only")
    #test_file = st.text_input("Test set", placeholder=".csv, .txt only")
    datasets_folder = 'Datasets'

    # List all files in the datasets folder (only CSV and Excel files)
    dataset_files = [f for f in os.listdir(datasets_folder) if f.endswith(('.csv', '.xlsx'))]

    dataset_files.insert(0, "Bir veri seti seçiniz")
    # Create a selectbox for choosing a dataset
    selected_file = st.selectbox('Veri setleri:', dataset_files)

    # If a dataset is selected, display it
    if selected_file!="Bir veri seti seçiniz":
        file_path = os.path.join(datasets_folder, selected_file)
        
        # Load the dataset
        if selected_file.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif selected_file.endswith('.xlsx'):
            df = pd.read_excel(file_path)

        # Display the dataset in Streamlit
        st.write(f"### {selected_file}")
        filtered_df = dataframe_explorer(df) 
        st.dataframe(filtered_df, use_container_width=True) 
        btn_describe=st.button("Veri setini özetle (describe)")
        
        # Add a button to show the describe() results
        if btn_describe:
            # Display describe() result
            st.write("### Veri seti istatistiği (describe):")
            st.write(df.describe())
            hide_summary = st.button("Gizle")
            if hide_summary:
                st.write("")
        if st.button("Veri tiplerini göster (info)"):
            # Display describe() result
            st.write("### Veri tipleri (info):")
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            hide_summary = st.button("Gizle")
            if hide_summary:
                st.text("")


        
    

    # Sidebar menu for selecting a specific regression model
model_menu = ['Model Seç', 'Nearest Neighbors', 'Linear Regression', 'SVR (Support Vector Regression)', 'Random Forest Regressor']
    
    # Create a sidebar dropdown or radio button for model selection
selected_model = st.sidebar.selectbox("Bir Regresyon Modeli Seçiniz:", model_menu)

    # Display content based on model selection
if selected_model == 'Model Seç':
        st.write("Daha fazla bilgi edinmek için lütfen sol taraftaki açılır menüden bir model seçin.")

elif selected_model == 'Nearest Neighbors':
        display_nearest_neighbors()

elif selected_model == 'Linear Regression':
        display_linear_regression()

elif selected_model == 'SVR (Support Vector Regression)':
        display_svr()

elif selected_model == 'Random Forest Regressor':
        display_random_forest()



# Display the selected page
