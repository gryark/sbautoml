import streamlit as st

def render_linear():
    st.subheader("Linear Regression")
    st.write("""
    Doğrusal Regresyon, bağımlı bir değişken ile bir veya daha fazla bağımsız değişken arasındaki ilişkiyi doğrusal bir denklemi uydurarak modellemektedir.
    
    - Basit ve yaygın olarak kullanılan bir regresyon tekniğidir.
    - En iyi uyumlu çizgiyi bulmak için kareli kalıntıların toplamını minimize etmeyi amaçlar.
    
    ## Ana Formül:
    - `y = mx + b`
    Nerede:
    - `m` eğim (katsayı),
    - `b` kesişim noktasını temsil eder.
    """)

if __name__ == "__main__":
    render_linear()